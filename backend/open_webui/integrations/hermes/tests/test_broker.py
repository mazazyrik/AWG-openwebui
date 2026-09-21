from types import SimpleNamespace
from unittest.mock import AsyncMock
import base64
import hashlib
import json

import pytest
from fastapi import HTTPException
from open_webui.integrations.hermes import broker, identity


class RedisStub:
    def __init__(self, values=None):
        self.values = values or {}

    async def get(self, key):
        return self.values.get(key)

    async def rpush(self, key, value):
        self.values.setdefault(key, []).append(value)

    async def expire(self, key, ttl):
        return True

    async def zadd(self, key, mapping):
        self.values.setdefault(key, {}).update(mapping)

    async def zrem(self, key, *members):
        values = self.values.setdefault(key, {})
        for member in members:
            values.pop(member.decode() if isinstance(member, bytes) else member, None)

    async def zrangebyscore(self, key, minimum, maximum):
        return [member for member, score in self.values.get(key, {}).items() if score <= float(maximum)]


def qwen_request(redis, payload, *, token='qwen-token', run_id='run-a', scope_id='scope-a', model='awg-qwen'):
    async def request_json():
        return payload

    return (
        SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(redis=redis)),
            json=request_json,
        ),
        f'Bearer {token}',
        run_id,
        scope_id,
        model,
    )


@pytest.mark.asyncio
async def test_broker_rejects_revoked_capability(monkeypatch):
    monkeypatch.setattr(identity, 'HERMES_CONTROL_SECRET', 'test-control-secret-that-is-long-enough')
    token, _ = identity.issue_principal_token(
        'user-a', chat_id='chat-a', message_id='message-a', run_id='run-a', allowed_file_ids=[]
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(redis=RedisStub())))

    with pytest.raises(HTTPException) as error:
        await broker.get_hermes_principal(request, f'Bearer {token}')

    assert error.value.status_code == 401
    assert error.value.detail == 'Hermes capability is inactive'


@pytest.mark.asyncio
async def test_attachment_capability_is_exact_allowlist():
    principal = SimpleNamespace(scope_id='scope-a', allowed_file_ids=frozenset({'file-a'}))
    user = SimpleNamespace(id='user-a')
    request = SimpleNamespace()

    async def request_json():
        return {'file_id': 'file-b'}

    request.json = request_json
    with pytest.raises(HTTPException) as error:
        await broker.call_tool('read_attachment', request, (principal, user))

    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_temporary_chat_cannot_stage_plugin(monkeypatch):
    principal = SimpleNamespace(scope_id='scope-a', chat_id='local', allowed_file_ids=frozenset())
    request = SimpleNamespace()

    async def request_json():
        return {'name': 'safe-plugin', 'files': {'plugin.yaml': 'name: safe-plugin', '__init__.py': ''}}

    request.json = request_json
    monkeypatch.setattr(broker, 'is_saved_chat_id', lambda _: False)
    with pytest.raises(HTTPException) as error:
        await broker.call_tool('stage_plugin', request, (principal, SimpleNamespace(id='user-a')))

    assert error.value.status_code == 403


@pytest.mark.asyncio
async def test_artifact_is_staged_without_registration_or_ui_linkage(monkeypatch):
    redis = RedisStub()
    request = SimpleNamespace(
        headers={'content-length': '4'},
        app=SimpleNamespace(state=SimpleNamespace(redis=redis)),
    )

    async def body():
        return b'%PDF'

    request.body = body
    monkeypatch.setattr(broker.Storage, 'upload_file', lambda *args: ('stored', '/tmp/stored'), raising=False)
    monkeypatch.setattr(broker.Files, 'insert_new_file', AsyncMock())
    monkeypatch.setattr(broker.Chats, 'insert_chat_files', AsyncMock())
    monkeypatch.setattr(broker.Chats, 'add_message_files_by_id_and_message_id', AsyncMock())
    monkeypatch.setattr(broker, 'publish_event', AsyncMock())
    monkeypatch.setattr(broker, 'is_saved_chat_id', lambda _: True)
    principal = SimpleNamespace(scope_id='scope-a', jti='jti-a', chat_id='chat-a', message_id='message-a')
    user = SimpleNamespace(id='user-a')

    filename = base64.urlsafe_b64encode('Отчёт.pdf'.encode()).decode()
    result = await broker.upload_artifact(request, filename, (principal, user))

    assert result['status'] == 'staged'
    assert result['filename'] == 'Отчёт.pdf'
    assert redis.values['awg:hermes:artifacts:jti-a']
    assert redis.values[broker.STAGED_ARTIFACT_INDEX]
    broker.Files.insert_new_file.assert_not_awaited()
    broker.Chats.insert_chat_files.assert_not_awaited()
    broker.Chats.add_message_files_by_id_and_message_id.assert_not_awaited()
    broker.publish_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_validated_artifact_is_registered_linked_and_emitted(monkeypatch):
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(redis=RedisStub())))
    user = SimpleNamespace(id='user-a')
    file = SimpleNamespace(id='file-a', filename='report.pdf', meta={'content_type': 'application/pdf'})
    monkeypatch.setattr(broker.Files, 'insert_new_file', AsyncMock(return_value=file))
    monkeypatch.setattr(broker.Chats, 'insert_chat_files', AsyncMock())
    monkeypatch.setattr(broker.Chats, 'add_message_files_by_id_and_message_id', AsyncMock())
    monkeypatch.setattr(broker, 'publish_event', AsyncMock())
    monkeypatch.setattr(broker, 'is_saved_chat_id', lambda _: True)
    staged = [
        {
            'artifact_id': 'artifact-a',
            'filename': 'report.pdf',
            'path': '/tmp/staged.pdf',
            'hash': 'hash-a',
            'size': 4,
            'content_type': 'application/pdf',
            'scope_id': 'scope-a',
            'user_id': 'user-a',
            'chat_id': 'chat-a',
            'message_id': 'message-a',
        }
    ]

    files = await broker.commit_staged_artifacts(request, user, staged)

    assert files[0]['id'] == 'file-a'
    broker.Chats.insert_chat_files.assert_awaited_once_with('chat-a', 'message-a', ['file-a'], 'user-a')
    broker.Chats.add_message_files_by_id_and_message_id.assert_awaited_once()
    broker.publish_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_expired_staged_artifact_is_deleted_without_run_key(monkeypatch):
    redis = RedisStub()
    artifact = {'artifact_id': 'old', 'path': '/tmp/hermes-staged-old.pdf'}
    member = broker.JSONCodec.dumps(artifact, separators=(',', ':'))
    redis.values[broker.STAGED_ARTIFACT_INDEX] = {member: 1}
    deleted = []
    monkeypatch.setattr(broker.Storage, 'delete_file', lambda path: deleted.append(path), raising=False)

    count = await broker.reap_expired_staged_artifacts(redis)

    assert count == 1
    assert deleted == ['/tmp/hermes-staged-old.pdf']
    assert redis.values[broker.STAGED_ARTIFACT_INDEX] == {}


def test_archive_guard_rejects_excessive_compression_ratio(tmp_path):
    import zipfile

    path = tmp_path / 'bomb.docx'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('word/document.xml', 'A' * 1_000_000)

    with pytest.raises(ValueError, match='compression ratio'):
        broker._validate_archive(str(path))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('run_id', 'scope_id', 'model'),
    [('run-b', 'scope-a', 'awg-qwen'), ('run-a', 'scope-b', 'awg-qwen'), ('run-a', 'scope-a', 'other')],
)
async def test_qwen_proxy_rejects_wrong_run_scope_or_model(monkeypatch, run_id, scope_id, model):
    token = 'qwen-token'
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    redis = RedisStub(
        {f'awg:hermes:qwen:{token_hash}': json.dumps({'scope_id': 'scope-a', 'run_id': 'run-a', 'model': 'awg-qwen'})}
    )
    monkeypatch.setattr(broker, 'HERMES_QWEN_API_KEY', 'server-secret')
    monkeypatch.setattr(broker, 'HERMES_QWEN_MODEL', 'fixed-qwen')
    request, authorization, _, _, _ = qwen_request(
        redis, {'model': 'awg-qwen', 'messages': []}, token=token, run_id=run_id, scope_id=scope_id, model=model
    )

    with pytest.raises(HTTPException) as error:
        await broker.qwen_chat_completion(request, authorization, run_id, scope_id, model)

    assert error.value.status_code == 403


@pytest.mark.asyncio
async def test_qwen_proxy_rejects_expired_or_revoked_token(monkeypatch):
    monkeypatch.setattr(broker, 'HERMES_QWEN_API_KEY', 'server-secret')
    monkeypatch.setattr(broker, 'HERMES_QWEN_MODEL', 'fixed-qwen')
    request, authorization, run_id, scope_id, model = qwen_request(RedisStub(), {'model': 'awg-qwen', 'messages': []})

    with pytest.raises(HTTPException) as error:
        await broker.qwen_chat_completion(request, authorization, run_id, scope_id, model)

    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_qwen_proxy_rejects_arbitrary_model_and_routing(monkeypatch):
    token = 'qwen-token'
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    redis = RedisStub(
        {f'awg:hermes:qwen:{token_hash}': json.dumps({'scope_id': 'scope-a', 'run_id': 'run-a', 'model': 'awg-qwen'})}
    )
    monkeypatch.setattr(broker, 'HERMES_QWEN_API_KEY', 'server-secret')
    monkeypatch.setattr(broker, 'HERMES_QWEN_MODEL', 'fixed-qwen')
    request, authorization, run_id, scope_id, model = qwen_request(
        redis, {'model': 'other', 'messages': [], 'base_url': 'https://example.invalid'}, token=token
    )

    with pytest.raises(HTTPException) as error:
        await broker.qwen_chat_completion(request, authorization, run_id, scope_id, model)

    assert error.value.status_code == 400
    paths = {route.path for route in broker.router.routes}
    assert '/qwen/v1/chat/completions' in paths
    assert not any(path.startswith('/qwen/') and path != '/qwen/v1/chat/completions' for path in paths)
