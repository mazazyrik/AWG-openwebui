import asyncio
from unittest.mock import AsyncMock

import pytest
from open_webui.utils.json_codec import JSONCodec
from open_webui.utils.mcp import tool_cache

SPECS = [{'name': 'search', 'parameters': {'type': 'object'}}]
CONNECTION = {
    'type': 'mcp',
    'url': 'https://mcp.example.test',
    'auth_type': 'bearer',
    'headers': {'Authorization': 'static-secret'},
    'info': {'id': 'server-1'},
}


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.set_calls = []
        self.deleted = []
        self.fail_get = False
        self.fail_cache_set = False

    async def get(self, key):
        if self.fail_get:
            raise ConnectionError('redis unavailable')
        return self.values.get(key)

    async def set(self, key, value, **kwargs):
        self.set_calls.append((key, value, kwargs))
        if self.fail_cache_set and not key.endswith(':lock'):
            raise ConnectionError('redis unavailable')
        if kwargs.get('nx') and key in self.values:
            return False
        self.values[key] = value
        return True

    async def eval(self, script, numkeys, key, token):
        if self.values.get(key) == token:
            del self.values[key]
            return 1
        return 0

    async def delete(self, *keys):
        self.deleted.extend(keys)
        for key in keys:
            self.values.pop(key, None)


@pytest.mark.asyncio
async def test_cache_miss_loads_and_stores_with_short_expiry():
    redis = FakeRedis()
    loader = AsyncMock(return_value=SPECS)

    specs, cache_hit = await tool_cache.get_mcp_tool_specs(redis, CONNECTION, loader)

    cache_writes = [call for call in redis.set_calls if not call[0].endswith(':lock')]
    assert specs == SPECS
    assert cache_hit is False
    loader.assert_awaited_once()
    assert len(cache_writes) == 1
    key, value, options = cache_writes[0]
    assert options == {'ex': tool_cache.MCP_TOOL_SPECS_CACHE_TTL}
    assert options['ex'] == 30
    assert JSONCodec.loads(value) == SPECS
    assert 'static-secret' not in key
    assert 'static-secret' not in value


@pytest.mark.asyncio
async def test_cache_hit_skips_loader():
    redis = FakeRedis()
    key = tool_cache._cache_key(CONNECTION)
    redis.values[key] = JSONCodec.dumps(SPECS)
    loader = AsyncMock()

    specs, cache_hit = await tool_cache.get_mcp_tool_specs(redis, CONNECTION, loader)

    assert specs == SPECS
    assert cache_hit is True
    loader.assert_not_awaited()


@pytest.mark.asyncio
async def test_cache_read_error_fails_open_without_locking():
    redis = FakeRedis()
    redis.fail_get = True
    loader = AsyncMock(return_value=SPECS)

    result = await tool_cache.get_mcp_tool_specs(redis, CONNECTION, loader)

    assert result == (SPECS, False)
    loader.assert_awaited_once()
    assert redis.set_calls == []


@pytest.mark.asyncio
async def test_cache_write_error_returns_loaded_specs():
    redis = FakeRedis()
    redis.fail_cache_set = True
    loader = AsyncMock(return_value=SPECS)

    result = await tool_cache.get_mcp_tool_specs(redis, CONNECTION, loader)

    assert result == (SPECS, False)
    loader.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'connection',
    [
        {**CONNECTION, 'auth_type': 'session'},
        {**CONNECTION, 'auth_type': 'oauth_2.1'},
        {**CONNECTION, 'auth_type': 'oauth_2.1_static'},
        {**CONNECTION, 'headers': {'X-User': '{{USER_ID}}'}},
        {**CONNECTION, 'forward_cookies': True},
    ],
)
async def test_dynamic_or_user_dependent_connection_bypasses_cache(connection):
    redis = FakeRedis()
    loader = AsyncMock(return_value=SPECS)

    result = await tool_cache.get_mcp_tool_specs(redis, connection, loader)

    assert result == (SPECS, False)
    loader.assert_awaited_once()
    assert redis.set_calls == []


@pytest.mark.asyncio
async def test_forwarded_user_headers_bypass_cache(monkeypatch):
    monkeypatch.setattr(tool_cache, 'ENABLE_FORWARD_USER_INFO_HEADERS', True)
    redis = FakeRedis()
    loader = AsyncMock(return_value=SPECS)

    result = await tool_cache.get_mcp_tool_specs(redis, CONNECTION, loader)

    assert result == (SPECS, False)
    loader.assert_awaited_once()
    assert redis.set_calls == []


def test_cache_key_changes_when_connection_config_changes():
    changed = {**CONNECTION, 'url': 'https://other.example.test'}

    assert tool_cache._cache_key(CONNECTION) != tool_cache._cache_key(changed)


@pytest.mark.asyncio
async def test_invalidation_deletes_only_mcp_config_keys():
    redis = FakeRedis()
    changed = {**CONNECTION, 'url': 'https://other.example.test'}

    await tool_cache.invalidate_mcp_tool_specs_cache(redis, [CONNECTION, changed, {'type': 'openapi'}])

    assert set(redis.deleted) == {tool_cache._cache_key(CONNECTION), tool_cache._cache_key(changed)}


@pytest.mark.asyncio
async def test_concurrent_miss_shares_loader_result():
    redis = FakeRedis()
    entered_loader = asyncio.Event()
    release_loader = asyncio.Event()
    loader_calls = 0

    async def loader():
        nonlocal loader_calls
        loader_calls += 1
        entered_loader.set()
        await release_loader.wait()
        return SPECS

    first = asyncio.create_task(tool_cache.get_mcp_tool_specs(redis, CONNECTION, loader))
    await entered_loader.wait()
    second = asyncio.create_task(tool_cache.get_mcp_tool_specs(redis, CONNECTION, loader))
    await asyncio.sleep(0.05)
    release_loader.set()

    first_result, second_result = await asyncio.gather(first, second)

    assert first_result == (SPECS, False)
    assert second_result == (SPECS, True)
    assert loader_calls == 1
