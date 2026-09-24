import asyncio
import hashlib
import json
import logging
import uuid
from collections.abc import Awaitable, Callable

from open_webui.env import ENABLE_FORWARD_USER_INFO_HEADERS, REDIS_KEY_PREFIX
from open_webui.utils.json_codec import JSONCodec

log = logging.getLogger(__name__)

MCP_TOOL_SPECS_CACHE_TTL = 30
MCP_TOOL_SPECS_LOCK_TTL = 10
MCP_TOOL_SPECS_LOCK_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


def mcp_tool_cache_label(server_id: str) -> str:
    return hashlib.sha256(server_id.encode()).hexdigest()[:12]


def _cache_key(connection: dict) -> str:
    server_id = (connection.get('info') or {}).get('id', '')
    serialized = json.dumps(connection, sort_keys=True, separators=(',', ':'), default=str)
    server_hash = mcp_tool_cache_label(str(server_id))
    config_hash = hashlib.sha256(serialized.encode()).hexdigest()
    return f'{REDIS_KEY_PREFIX}:mcp:tool_specs:{server_hash}:{config_hash}'


def _is_cacheable(connection: dict) -> bool:
    auth_type = connection.get('auth_type', 'bearer')
    if auth_type not in (None, 'none', 'bearer'):
        return False
    if connection.get('forward_cookies') or ENABLE_FORWARD_USER_INFO_HEADERS:
        return False

    headers = connection.get('headers')
    if isinstance(headers, dict):
        return not any('{{' in str(value) for value in headers.values())
    return not (isinstance(headers, str) and '{{' in headers)


async def _read_cache(redis, key: str) -> tuple[list[dict] | None, bool]:
    try:
        value = await redis.get(key)
    except Exception as exc:
        log.debug('MCP tool spec cache read failed (%s)', type(exc).__name__)
        return None, False

    if value is None:
        return None, True

    try:
        specs = JSONCodec.loads(value)
    except Exception as exc:
        log.debug('MCP tool spec cache decode failed (%s)', type(exc).__name__)
        return None, True

    if isinstance(specs, list) and all(
        isinstance(spec, dict) and 'name' in spec and 'parameters' in spec for spec in specs
    ):
        return specs, True
    return None, True


async def _wait_for_cached_specs(redis, key: str) -> list[dict] | None:
    for _ in range(5):
        await asyncio.sleep(0.05)
        cached_specs, available = await _read_cache(redis, key)
        if cached_specs is not None:
            return cached_specs
        if not available:
            break
    return None


async def _load_and_cache(redis, key: str, loader: Callable[[], Awaitable[list[dict]]]) -> list[dict]:
    specs = await loader()
    try:
        await redis.set(key, JSONCodec.dumps(specs), ex=MCP_TOOL_SPECS_CACHE_TTL)
    except Exception as exc:
        log.debug('MCP tool spec cache write failed (%s)', type(exc).__name__)
    return specs


async def _load_with_lock(
    redis,
    key: str,
    lock_key: str,
    lock_token: str,
    loader: Callable[[], Awaitable[list[dict]]],
) -> tuple[list[dict], bool]:
    try:
        cached_specs, available = await _read_cache(redis, key)
        if cached_specs is not None:
            return cached_specs, True
        if not available:
            return await loader(), False
        return await _load_and_cache(redis, key, loader), False
    finally:
        try:
            await redis.eval(MCP_TOOL_SPECS_LOCK_RELEASE_SCRIPT, 1, lock_key, lock_token)
        except Exception as exc:
            log.debug('MCP tool spec cache lock release failed (%s)', type(exc).__name__)


async def get_mcp_tool_specs(
    redis,
    connection: dict,
    loader: Callable[[], Awaitable[list[dict]]],
) -> tuple[list[dict], bool]:
    if redis is None or not _is_cacheable(connection):
        return await loader(), False

    try:
        key = _cache_key(connection)
    except Exception:
        return await loader(), False

    cached_specs, available = await _read_cache(redis, key)
    if cached_specs is not None:
        return cached_specs, True
    if not available:
        return await loader(), False

    lock_key = f'{key}:lock'
    lock_token = uuid.uuid4().hex
    try:
        lock_acquired = await redis.set(lock_key, lock_token, nx=True, ex=MCP_TOOL_SPECS_LOCK_TTL)
    except Exception as exc:
        log.debug('MCP tool spec cache lock failed (%s)', type(exc).__name__)
        return await loader(), False

    if not lock_acquired:
        cached_specs = await _wait_for_cached_specs(redis, key)
        if cached_specs is not None:
            return cached_specs, True
        return await loader(), False

    return await _load_with_lock(redis, key, lock_key, lock_token, loader)


async def invalidate_mcp_tool_specs_cache(redis, connections: list[dict]) -> None:
    if redis is None:
        return

    try:
        keys = {
            _cache_key(connection)
            for connection in connections
            if isinstance(connection, dict) and connection.get('type', 'openapi') == 'mcp'
        }
        if not keys:
            return
        await redis.delete(*keys)
    except Exception as exc:
        log.warning('MCP tool spec cache invalidation failed (%s)', type(exc).__name__)
