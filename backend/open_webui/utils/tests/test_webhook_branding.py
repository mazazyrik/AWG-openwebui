from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from open_webui.utils import webhook


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('favicon', 'base_url', 'expected'),
    [
        ('/static/favicon.png', 'https://gpt.awg.test', 'https://gpt.awg.test/static/favicon.png'),
        ('/static/favicon.png', 'https://gpt.awg.test/', 'https://gpt.awg.test/static/favicon.png'),
        ('https://cdn.awg.test/custom.png', '', 'https://cdn.awg.test/custom.png'),
        ('/static/favicon.png', '', None),
        ('/static/favicon.png', None, None),
        ('/static/favicon.png', 'invalid', None),
    ],
)
async def test_teams_brand_image_is_publicly_addressable(favicon, base_url, expected):
    response = MagicMock()
    response.text = AsyncMock(return_value='ok')
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.post.return_value = response
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch.object(webhook, 'WEBUI_FAVICON_URL', favicon),
        patch.object(webhook.Config, 'get', AsyncMock(return_value=base_url)),
        patch.object(webhook, 'validate_url'),
        patch.object(webhook, 'get_ssrf_safe_session', return_value=session),
    ):
        result = await webhook.post_webhook(
            'AWG GPT', 'https://webhook.office.com/test', 'Test notification', {'event': 'chat.completed'}
        )

    assert result is True
    payload = session.post.call_args.kwargs['json']
    assert payload['summary'] == 'Test notification'
    assert payload['sections'][0].get('activityImage') == expected
    if expected is None:
        assert 'activityImage' not in payload['sections'][0]
