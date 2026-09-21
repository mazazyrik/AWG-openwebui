import pytest
from fastapi import HTTPException
from open_webui.integrations.hermes import identity
from open_webui.integrations.hermes.policy import TOOL_SCHEMAS, authorize_tool


@pytest.fixture(autouse=True)
def control_secret(monkeypatch):
    monkeypatch.setattr(identity, 'HERMES_CONTROL_SECRET', 'test-control-secret-that-is-long-enough')


def test_principal_token_binds_scope_run_files_and_ttl():
    token, jti = identity.issue_principal_token(
        'user-a',
        chat_id='chat-a',
        message_id='message-a',
        run_id='run-a',
        allowed_file_ids=['file-b', 'file-a', 'file-a'],
        now=100,
    )

    principal = identity.verify_principal_token(token, now=101)

    assert principal.user_id == 'user-a'
    assert principal.scope_id == identity.user_scope_id('user-a')
    assert principal.jti == jti
    assert principal.run_id == 'run-a'
    assert principal.allowed_file_ids == frozenset({'file-a', 'file-b'})
    assert principal.expires_at == 100 + identity.TOKEN_TTL_SECONDS


def test_principal_token_rejects_tamper_and_expiry():
    token, _ = identity.issue_principal_token(
        'user-a',
        chat_id='chat-a',
        message_id='message-a',
        run_id='run-a',
        allowed_file_ids=[],
        now=100,
    )

    with pytest.raises(HTTPException) as tampered:
        identity.verify_principal_token(token + 'x', now=101)
    with pytest.raises(HTTPException) as expired:
        identity.verify_principal_token(token, now=101 + identity.TOKEN_TTL_SECONDS)

    assert tampered.value.status_code == 401
    assert expired.value.status_code == 401


@pytest.mark.parametrize('tool', TOOL_SCHEMAS)
def test_tool_schemas_deny_additional_properties(tool):
    assert TOOL_SCHEMAS[tool]['additionalProperties'] is False


def test_policy_denies_unknown_and_invalid_arguments():
    assert authorize_tool('scope-a', 'shell', {}).reason == 'unknown_tool'
    assert authorize_tool('scope-a', 'search_confluence', {'query': 'x', 'extra': True}).reason == 'invalid_arguments'
    assert authorize_tool('scope-a', 'read_attachment', {'file_id': '../other'}).reason == 'invalid_arguments'
    assert (
        authorize_tool('scope-a', 'stage_plugin', {'name': 'Bad', 'files': {'a': '', 'b': ''}}).reason
        == 'invalid_arguments'
    )
