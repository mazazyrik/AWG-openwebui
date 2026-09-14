import httpx
import pytest
from open_webui.integrations.confluence import client as client_module
from open_webui.integrations.confluence.client import (
    API_TOKEN_ENV,
    API_URL_ENV,
    MCP_URL_ENV,
    ConfluenceAPIClient,
    ConfluenceMCPClient,
)

ORIGINAL_ASYNC_CLIENT = httpx.AsyncClient


def _async_client_factory(handler):
    def factory(**kwargs):
        return ORIGINAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler))

    return factory


@pytest.mark.asyncio
async def test_api_client_checks_health_and_queues_sync(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == '/health':
            return httpx.Response(200, json={'ok': True, 'enabled': True})
        assert request.headers['Authorization'] == 'Bearer admin-secret'
        return httpx.Response(
            200,
            json={'run_id': 'remote-run', 'status': 'queued', 'mode': 'full'},
        )

    monkeypatch.setenv(API_URL_ENV, 'http://confluence-rag.test')
    monkeypatch.setenv(API_TOKEN_ENV, 'admin-secret')
    monkeypatch.setattr(
        client_module.httpx,
        'AsyncClient',
        _async_client_factory(handler),
    )

    client = ConfluenceAPIClient()
    assert await client.check() == {'ok': True, 'enabled': True}
    run = await client.trigger_sync('full')
    assert run['run_id'] == 'remote-run'


class RestrictionsClient(ConfluenceMCPClient):
    def __init__(self, restrictions: object) -> None:
        super().__init__()
        self.restrictions = restrictions

    async def _call(self, client, name, arguments):
        return self.restrictions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('restrictions', 'expected'),
    [
        ({'read': {'users': [], 'groups': []}}, True),
        ({'read': {'users': ['restricted-user'], 'groups': []}}, False),
        ({'read': {}}, False),
        (None, False),
    ],
)
async def test_mcp_page_access_fails_closed(monkeypatch, restrictions, expected) -> None:
    monkeypatch.setenv(MCP_URL_ENV, 'http://mcp.test/mcp')
    monkeypatch.setattr(
        client_module.httpx,
        'AsyncClient',
        _async_client_factory(lambda _: httpx.Response(200)),
    )
    client = RestrictionsClient(restrictions)

    assert await client.can_read_page('123') is expected


def test_mcp_sse_response_parser() -> None:
    payload = ConfluenceMCPClient._response_payload('event: message\ndata: {"jsonrpc":"2.0","result":{"tools":[]}}')
    assert payload['result'] == {'tools': []}
