from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from open_webui.integrations.hermes.client import HermesClient, HermesRuntime
from open_webui.integrations.hermes.identity import user_scope_id
from open_webui.utils.json_codec import JSONCodec


class PipelineStub:
    def __init__(self, redis):
        self.redis = redis
        self.commands = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    def lrange(self, key, start, end):
        self.commands.append(('lrange', key))

    def delete(self, key):
        self.commands.append(('delete', key))

    async def execute(self):
        key = self.commands[0][1]
        values = list(self.redis.lists.get(key, []))
        self.redis.lists.pop(key, None)
        return values, 1


class RedisStub:
    def __init__(self):
        self.values = {}
        self.lists = {}
        self.expiries = {}

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.values:
            return False
        self.values[key] = value
        self.expiries[key] = ex
        return True

    async def eval(self, script, count, key, owner, *args):
        if self.values.get(key) != owner:
            return 0
        if 'expire' in script:
            return 1
        self.values.pop(key, None)
        return 1

    def pipeline(self, transaction=True):
        return PipelineStub(self)


def make_client(user_id, redis):
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(redis=redis)))
    return HermesClient(request, SimpleNamespace(id=user_id), {}, {'id': 'awg-qwen'})


def test_grounded_instructions_include_source_ids_and_exact_citation_format():
    client = make_client('user-a', RedisStub())
    state = SimpleNamespace(
        route='confluence_grounded',
        sources=(
            {
                'id': 'S1',
                'title': 'Project page',
                'url': 'https://conf.awg.ru/pages/1',
                'page_id': '1',
                'version': 2,
                'text': 'Verified project fact.',
            },
        ),
    )

    instructions = client._instructions(state, [], True)

    assert '"id":"S1"' in instructions
    assert 'Verified project fact.' in instructions
    assert '`[S<number>] <matching source URL>`' in instructions


@pytest.mark.asyncio
async def test_same_user_lease_serializes_and_release_is_owner_checked(monkeypatch):
    redis = RedisStub()
    first = make_client('user-a', redis)
    second = make_client('user-a', redis)

    await first._acquire_lease()
    monkeypatch.setattr('open_webui.integrations.hermes.client.HERMES_REQUEST_TIMEOUT', 0)
    with pytest.raises(HTTPException) as error:
        await second._acquire_lease()
    assert error.value.status_code == 409

    await second._release_lease()
    assert redis.values[f'awg:hermes:lease:{user_scope_id("user-a")}'] == first.lease_id
    await first._release_lease()
    assert f'awg:hermes:lease:{user_scope_id("user-a")}' not in redis.values


@pytest.mark.asyncio
async def test_different_users_have_independent_leases():
    redis = RedisStub()
    first = make_client('user-a', redis)
    second = make_client('user-b', redis)

    await first._acquire_lease()
    await second._acquire_lease()

    assert len(redis.values) == 2


@pytest.mark.asyncio
async def test_lease_renewal_fails_after_ownership_is_lost():
    redis = RedisStub()
    client = make_client('user-a', redis)
    await client._acquire_lease()
    redis.values[f'awg:hermes:lease:{user_scope_id("user-a")}'] = 'other-run'

    with pytest.raises(RuntimeError, match='lease was lost'):
        await client._renew_lease()


@pytest.mark.asyncio
async def test_artifacts_are_consumed_atomically():
    redis = RedisStub()
    client = make_client('user-a', redis)
    client.principal_jti = 'jti-a'
    key = 'awg:hermes:artifacts:jti-a'
    redis.lists[key] = [JSONCodec.dumps({'id': 'file-a'}), JSONCodec.dumps({'id': 'file-b'})]

    assert await client._consume_artifacts() == [{'id': 'file-a'}, {'id': 'file-b'}]
    assert await client._consume_artifacts() == []


@pytest.mark.asyncio
async def test_cleanup_removes_only_ephemeral_runtime(monkeypatch):
    redis = RedisStub()
    client = make_client('user-a', redis)
    calls = []

    async def control(method, path, payload):
        calls.append((method, path, payload))
        return {}

    monkeypatch.setattr(client, '_control_request', control)
    client.runtime = HermesRuntime('http://runtime', 'key', 'persistent', False)
    await client.cleanup()
    client.runtime = HermesRuntime('http://runtime', 'key', 'temporary', True)
    await client.cleanup()

    assert calls == [('POST', '/v1/runtimes/temporary/delete', {})]


@pytest.mark.asyncio
async def test_provision_issues_scoped_short_lived_qwen_capability(monkeypatch):
    redis = RedisStub()
    client = make_client('user-a', redis)
    client.metadata = {'chat_id': 'chat-a', 'message_id': 'message-a'}
    calls = []

    async def control(method, path, payload):
        calls.append((method, path, payload))
        if path == '/v1/runtimes/ensure':
            return {
                'base_url': 'http://runtime',
                'api_key': 'runtime-key',
                'runtime_id': 'runtime-a',
                'ephemeral': False,
            }
        return {'status': 'active'}

    monkeypatch.setattr(client, '_control_request', control)
    monkeypatch.setattr(
        'open_webui.integrations.hermes.client.Memories.get_memories_by_user_id', AsyncMock(return_value=[])
    )

    await client.provision(ephemeral=False, allowed_file_ids=[], run_scope_id='run-a')

    qwen_key = next(key for key in redis.values if key.startswith('awg:hermes:qwen:'))
    assert JSONCodec.loads(redis.values[qwen_key]) == {
        'scope_id': user_scope_id('user-a'),
        'run_id': 'run-a',
        'model': 'awg-qwen',
    }
    assert redis.expiries[qwen_key] is not None
    capability = calls[-1][2]
    assert capability['qwen_run_id'] == 'run-a'
    assert capability['qwen_scope_id'] == user_scope_id('user-a')
