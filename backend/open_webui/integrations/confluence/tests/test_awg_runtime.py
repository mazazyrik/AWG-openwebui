import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from open_webui.integrations.confluence.grounding_filter import UNAVAILABLE, Filter
from open_webui.integrations.confluence.runtime import (
    MAX_PROVIDER_RESPONSE_BYTES,
    STATE_VERSION,
    AwgRequestState,
    AwgResponseRejected,
    attest_awg_attachment,
    clear_awg_request_state,
    extract_awg_provider_text,
    get_awg_request_state,
    register_awg_invocation,
    set_awg_request_state,
)
from starlette.responses import StreamingResponse

FILTER_ID = 'awg-filter'
MODEL = {'id': 'awg-gpt', 'info': {'meta': {'filterIds': [FILTER_ID]}}}


def make_state(invocation_id, **overrides):
    state = AwgRequestState(
        state_version=STATE_VERSION,
        route='confluence_grounded',
        model_id=MODEL['id'],
        invocation_id=invocation_id,
        filter_id=FILTER_ID,
        profile_version='2026-09-16.1',
        prompt_hash='hash',
        response_kind='grounded_fact',
        sources=(),
        memory_operation=None,
        scope_decision='test',
        unavailable=False,
        unavailable_reason=None,
        client_stream=False,
        provider_required=True,
        deterministic_answer=None,
    )
    return replace(state, **overrides)


def attested_request(*, stream=False):
    request = SimpleNamespace(state=SimpleNamespace())
    metadata = {}
    invocation_id = register_awg_invocation(request, MODEL, metadata)
    attest_awg_attachment(request, MODEL, metadata, FILTER_ID, stream)
    return request, metadata, invocation_id


@pytest.mark.parametrize(
    'invalid',
    [
        None,
        {'state_version': STATE_VERSION},
        make_state('wrong'),
        replace(make_state('wrong'), state_version=STATE_VERSION + 1),
        replace(make_state('wrong'), filter_id='other-filter'),
        replace(make_state('wrong'), response_kind='conversational'),
    ],
)
def test_attested_invalid_state_fails_closed(invalid):
    request, metadata, invocation_id = attested_request()
    set_awg_request_state(request, MODEL['id'], invocation_id, invalid)
    is_awg, state = get_awg_request_state(request, MODEL, metadata)
    assert is_awg is True
    assert state is None


def test_client_metadata_spoof_cannot_create_attestation():
    request = SimpleNamespace(state=SimpleNamespace())
    metadata = {'awg_invocation_id': 'spoof', 'filter_ids': [FILTER_ID]}
    assert get_awg_request_state(request, MODEL, metadata) == (False, None)


def test_attested_response_kind_must_match_route_contract():
    request, metadata, invocation_id = attested_request()
    invalid = make_state(invocation_id, response_kind='conversational')
    set_awg_request_state(request, MODEL['id'], invocation_id, invalid)
    assert get_awg_request_state(request, MODEL, metadata) == (True, None)


def test_duplicate_model_invocations_are_independent_and_cleanup_exact():
    request = SimpleNamespace(state=SimpleNamespace())
    first_metadata = {}
    second_metadata = {}
    first_id = register_awg_invocation(request, MODEL, first_metadata)
    second_id = register_awg_invocation(request, MODEL, second_metadata)
    attest_awg_attachment(request, MODEL, first_metadata, FILTER_ID, False)
    attest_awg_attachment(request, MODEL, second_metadata, FILTER_ID, True)
    set_awg_request_state(request, MODEL['id'], first_id, make_state(first_id))
    set_awg_request_state(request, MODEL['id'], second_id, make_state(second_id, client_stream=True))
    assert get_awg_request_state(request, MODEL, first_metadata)[1].invocation_id == first_id
    assert get_awg_request_state(request, MODEL, second_metadata)[1].invocation_id == second_id
    clear_awg_request_state(request, MODEL, first_metadata)
    assert get_awg_request_state(request, MODEL, first_metadata) == (False, None)
    assert get_awg_request_state(request, MODEL, second_metadata)[1].invocation_id == second_id


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_missing_attested_state_uses_original_response_contract(stream):
    request, metadata, _ = attested_request(stream=stream)
    body = {'messages': [{'role': 'assistant', 'content': 'unsafe'}]}
    result = await Filter().outlet(
        copy.deepcopy(body),
        __request__=request,
        __metadata__=metadata,
        __model__=MODEL,
        __id__=FILTER_ID,
    )
    assert result['messages'][0]['content'] == UNAVAILABLE


def sse_response(chunks):
    async def iterator():
        for chunk in chunks:
            yield chunk

    return StreamingResponse(iterator(), media_type='text/event-stream')


@pytest.mark.asyncio
async def test_sse_incremental_decoder_handles_split_utf8_and_json_lines():
    event = (
        'data: '
        + json.dumps({'choices': [{'delta': {'content': 'Привет'}}]}, ensure_ascii=False)
        + '\n\ndata: [DONE]\n\n'
    )
    raw = event.encode('utf-8')
    split = raw.index('П'.encode()) + 1
    assert (
        await extract_awg_provider_text(sse_response([raw[:split], raw[split : split + 5], raw[split + 5 :]]))
        == 'Привет'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('chunks', 'reason'),
    [
        ([b'data: {bad}\n\n'], 'provider_stream_json_invalid'),
        ([b'\xff'], 'provider_stream_utf8_invalid'),
        ([b'x' * (MAX_PROVIDER_RESPONSE_BYTES + 1)], 'provider_stream_too_large'),
    ],
)
async def test_sse_rejects_malformed_utf8_json_and_oversize(chunks, reason):
    with pytest.raises(AwgResponseRejected, match=reason):
        await extract_awg_provider_text(sse_response(chunks))
