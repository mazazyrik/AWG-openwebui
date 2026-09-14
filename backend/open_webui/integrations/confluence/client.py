import json
import os
from typing import Any

import httpx

MCP_URL_ENV = 'CONFLUENCE_MCP_URL'
MCP_TOKEN_ENV = 'CONFLUENCE_MCP_TOKEN'
API_URL_ENV = 'CONFLUENCE_RAG_API_URL'
API_TOKEN_ENV = 'CONFLUENCE_RAG_ADMIN_TOKEN'
QDRANT_URL_ENV = 'CONFLUENCE_QDRANT_URL'
QDRANT_API_KEY_ENV = 'CONFLUENCE_QDRANT_API_KEY'


class ConfluenceClientError(RuntimeError):
    pass


def secret_presence() -> dict[str, bool]:
    return {
        MCP_URL_ENV: bool(os.getenv(MCP_URL_ENV)),
        MCP_TOKEN_ENV: bool(os.getenv(MCP_TOKEN_ENV)),
        API_URL_ENV: bool(os.getenv(API_URL_ENV)),
        API_TOKEN_ENV: bool(os.getenv(API_TOKEN_ENV)),
        QDRANT_URL_ENV: bool(os.getenv(QDRANT_URL_ENV)),
        QDRANT_API_KEY_ENV: bool(os.getenv(QDRANT_API_KEY_ENV)),
    }


class ConfluenceAPIClient:
    def __init__(self) -> None:
        self.base_url = (os.getenv(API_URL_ENV) or '').rstrip('/')
        self.token = os.getenv(API_TOKEN_ENV) or ''

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    async def check(self) -> dict[str, Any]:
        if not self.configured:
            return {'ok': False, 'reason': 'api_not_configured'}
        try:
            async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
                response = await client.get(f'{self.base_url}/health')
                response.raise_for_status()
                payload = response.json()
            return {
                'ok': payload.get('ok') is True,
                'enabled': payload.get('enabled') is True,
            }
        except (httpx.HTTPError, ValueError):
            return {'ok': False, 'reason': 'api_unavailable'}

    async def trigger_sync(self, mode: str) -> dict[str, Any]:
        payload = await self._request(
            'POST',
            '/agent/confluence/sync',
            {'mode': mode},
        )
        if not isinstance(payload, dict) or not payload.get('run_id'):
            raise ConfluenceClientError('sync_response_invalid')
        return payload

    async def status(self) -> dict[str, Any] | None:
        payload = await self._request('GET', '/agent/confluence/status')
        if payload is not None and not isinstance(payload, dict):
            raise ConfluenceClientError('status_response_invalid')
        return payload

    async def _request(self, method: str, path: str, body: object = None) -> Any:
        if not self.configured:
            raise ConfluenceClientError('api_not_configured')
        try:
            async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
                response = await client.request(
                    method,
                    f'{self.base_url}{path}',
                    headers={'Authorization': f'Bearer {self.token}'},
                    json=body,
                )
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise ConfluenceClientError('api_unavailable') from error


class ConfluenceMCPClient:
    def __init__(self) -> None:
        self.url = os.getenv(MCP_URL_ENV) or ''
        self.token = os.getenv(MCP_TOKEN_ENV) or ''
        self._session_id: str | None = None
        self._request_id = 0

    @property
    def configured(self) -> bool:
        return bool(self.url)

    async def check(self) -> dict[str, Any]:
        if not self.configured:
            return {'ok': False, 'reason': 'mcp_not_configured'}
        try:
            async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
                tools = await self._list_tools(client)
            required = {'confluence_get_page_restrictions'}
            return {
                'ok': required.issubset(tools),
                'required_tools': sorted(required),
            }
        except ConfluenceClientError as error:
            return {'ok': False, 'reason': str(error)}

    async def can_read_page(self, page_id: str) -> bool:
        if not self.configured or not page_id:
            return False
        try:
            async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
                value = await self._call(
                    client,
                    'confluence_get_page_restrictions',
                    {'page_id': page_id},
                )
        except ConfluenceClientError:
            return False
        if not isinstance(value, dict) or not isinstance(value.get('read'), dict):
            return False
        read = value['read']
        users = read.get('users')
        groups = read.get('groups')
        return isinstance(users, list) and isinstance(groups, list) and not users and not groups

    async def _list_tools(self, client: httpx.AsyncClient) -> set[str]:
        await self._initialize(client)
        result = await self._post(client, 'tools/list', {})
        tools = result.get('tools')
        if not isinstance(tools, list):
            raise ConfluenceClientError('mcp_tools_invalid')
        return {str(tool['name']) for tool in tools if isinstance(tool, dict) and tool.get('name')}

    async def _call(
        self,
        client: httpx.AsyncClient,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        await self._initialize(client)
        result = await self._post(
            client,
            'tools/call',
            {'name': name, 'arguments': arguments},
        )
        if result.get('isError'):
            raise ConfluenceClientError('mcp_tool_failed')
        wrapped = result.get('structuredContent')
        raw = wrapped.get('result') if isinstance(wrapped, dict) else None
        if raw is None:
            content = result.get('content')
            if isinstance(content, list) and content and isinstance(content[0], dict):
                raw = content[0].get('text')
        if not isinstance(raw, str):
            raise ConfluenceClientError('mcp_tool_result_invalid')
        try:
            return json.loads(raw)
        except json.JSONDecodeError as error:
            raise ConfluenceClientError('mcp_tool_json_invalid') from error

    async def _initialize(self, client: httpx.AsyncClient) -> None:
        if self._session_id:
            return
        result, response = await self._post_response(
            client,
            'initialize',
            {
                'protocolVersion': '2025-06-18',
                'capabilities': {},
                'clientInfo': {'name': 'awg-openwebui', 'version': '1'},
            },
            include_session=False,
        )
        session_id = response.headers.get('mcp-session-id')
        if not session_id or 'serverInfo' not in result:
            raise ConfluenceClientError('mcp_initialize_failed')
        self._session_id = session_id
        response = await client.post(
            self.url,
            headers=self._headers(),
            json={'jsonrpc': '2.0', 'method': 'notifications/initialized'},
        )
        if response.status_code not in {200, 202}:
            raise ConfluenceClientError('mcp_notification_failed')

    async def _post(
        self,
        client: httpx.AsyncClient,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        result, _ = await self._post_response(client, method, params)
        return result

    async def _post_response(
        self,
        client: httpx.AsyncClient,
        method: str,
        params: dict[str, Any],
        *,
        include_session: bool = True,
    ) -> tuple[dict[str, Any], httpx.Response]:
        self._request_id += 1
        try:
            response = await client.post(
                self.url,
                headers=self._headers() if include_session else self._base_headers(),
                json={
                    'jsonrpc': '2.0',
                    'id': self._request_id,
                    'method': method,
                    'params': params,
                },
            )
            response.raise_for_status()
            payload = self._response_payload(response.text)
        except (httpx.HTTPError, ValueError) as error:
            raise ConfluenceClientError('mcp_unavailable') from error
        result = payload.get('result')
        if payload.get('error') or not isinstance(result, dict):
            raise ConfluenceClientError('mcp_result_invalid')
        return result, response

    def _headers(self) -> dict[str, str]:
        headers = self._base_headers()
        if self._session_id:
            headers['Mcp-Session-Id'] = self._session_id
        return headers

    def _base_headers(self) -> dict[str, str]:
        headers = {
            'Accept': 'application/json, text/event-stream',
            'Content-Type': 'application/json',
        }
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        return headers

    @staticmethod
    def _response_payload(value: str) -> dict[str, Any]:
        if 'data:' in value:
            value = value[value.find('data:') + len('data:') :].strip()
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError('invalid MCP response')
        return payload
