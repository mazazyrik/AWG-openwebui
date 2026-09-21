import io
import json
import os
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from open_webui.integrations.hermes import provisioner


def approved_manifest(runtime='registry.example/hermes@sha256:' + '2' * 64):
    return {
        'hermes_release': 'v2026.9.14',
        'hermes_upstream_commit': '345cd2b057a452236de401d3534b8502a7465e8d',
        'verified_at': '2026-09-21T00:00:00Z',
        'upstream_base_image': 'registry.example/upstream@sha256:' + '1' * 64,
        'runtime_image': runtime,
    }


def test_release_manifest_fails_closed_for_unverified_digest(tmp_path, monkeypatch):
    manifest = approved_manifest('registry.example/hermes:mutable')
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(provisioner, 'RELEASE_MANIFEST', str(path))
    monkeypatch.setattr(provisioner, 'RUNTIME_IMAGE', manifest['runtime_image'])

    with pytest.raises(RuntimeError, match='verified runtime digest'):
        provisioner._approved_runtime_image()


def test_release_manifest_requires_exact_release_commit_and_configured_image(tmp_path, monkeypatch):
    manifest = approved_manifest()
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(provisioner, 'RELEASE_MANIFEST', str(path))
    monkeypatch.setattr(provisioner, 'RUNTIME_IMAGE', manifest['runtime_image'])
    assert provisioner._approved_runtime_image() == manifest['runtime_image']

    monkeypatch.setattr(provisioner, 'RUNTIME_IMAGE', 'registry.example/other@sha256:' + '3' * 64)
    with pytest.raises(RuntimeError, match='not approved'):
        provisioner._approved_runtime_image()


def test_capability_archive_is_owned_by_runtime_user_and_mode_0600():
    archive = provisioner._archive({'principal-token': b'secret'})

    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        member = bundle.getmember('principal-token')
        assert member.mode == 0o600
        assert member.uid == 10000
        assert member.gid == 10000
        assert bundle.extractfile(member).read() == b'secret'


def test_runtime_config_restricts_api_server_toolsets():
    config = provisioner.RUNTIME_CONFIG.decode()

    assert 'platform_toolsets:\n  api_server:' in config
    assert '\ntoolsets:' not in config
    assert '    - mcp-awg' in config
    assert 'command: /opt/hermes/.venv/bin/python' in config
    assert 'AWG_HERMES_BROKER_URL: http://openwebui:8080' in config
    assert 'AWG_HERMES_PRINCIPAL_FILE: /run/awg-hermes/principal-token' in config


@pytest.mark.asyncio
async def test_gateway_identity_requires_uid_gid_10000(monkeypatch):
    docker = AsyncMock(return_value={'Processes': [['42', '10000', '10000', 'hermes gateway run']]})
    monkeypatch.setattr(
        provisioner,
        '_docker',
        docker,
    )
    await provisioner._verify_gateway_identity('container')
    assert docker.await_args.args[1].endswith('ps_args=-n%20-eo%20pid,uid,gid,args')

    monkeypatch.setattr(
        provisioner,
        '_docker',
        AsyncMock(return_value={'Processes': [['42', '0', '0', 'hermes gateway run']]}),
    )
    with pytest.raises(RuntimeError, match='UID/GID 10000'):
        await provisioner._verify_gateway_identity('container')


@pytest.mark.asyncio
async def test_readiness_requires_exact_builtin_and_awg_mcp_tools(monkeypatch):
    builtins = provisioner.ALLOWED_TOOLSETS - {'mcp-awg'}
    responses = {
        '/v1/capabilities': {
            'object': 'hermes.api_server.capabilities',
            'platform': 'hermes-agent',
            'endpoints': {'toolsets': {'method': 'GET', 'path': '/v1/toolsets'}},
        },
        '/v1/toolsets': {
            'object': 'list',
            'platform': 'api_server',
            'data': [{'name': name, 'enabled': True, 'tools': []} for name in builtins],
        },
        'discover_mcp_tools': sorted(
            {
                'mcp__awg__search_confluence',
                'mcp__awg__read_attachment',
                'mcp__awg__publish_artifact',
                'mcp__awg__stage_plugin',
            }
        ),
    }

    async def execute(container_id, command):
        source = ' '.join(command)
        if '/v1/capabilities' in source:
            return json.dumps(responses['/v1/capabilities'])
        if '/v1/toolsets' in source:
            return json.dumps(responses['/v1/toolsets'])
        if 'discover_mcp_tools' in source:
            return json.dumps(responses['discover_mcp_tools'])
        return 'ok'

    monkeypatch.setattr(provisioner, '_exec', execute)
    await provisioner._wait_ready('container')


@pytest.mark.asyncio
async def test_readiness_rejects_forbidden_web_toolset(monkeypatch):
    builtins = provisioner.ALLOWED_TOOLSETS - {'mcp-awg'}

    async def execute(container_id, command):
        source = ' '.join(command)
        if '/v1/capabilities' in source:
            return json.dumps(
                {
                    'object': 'hermes.api_server.capabilities',
                    'platform': 'hermes-agent',
                    'endpoints': {'toolsets': {'method': 'GET', 'path': '/v1/toolsets'}},
                }
            )
        if '/v1/toolsets' in source:
            data = [{'name': name, 'enabled': True, 'tools': []} for name in builtins]
            data.append({'name': 'web', 'enabled': True, 'tools': ['web_search']})
            return json.dumps({'object': 'list', 'platform': 'api_server', 'data': data})
        return 'ok'

    monkeypatch.setattr(provisioner, '_exec', execute)
    monkeypatch.setattr(provisioner.asyncio, 'sleep', AsyncMock())
    with pytest.raises(RuntimeError, match='approved toolsets'):
        await provisioner._wait_ready('container')


@pytest.mark.asyncio
async def test_background_plugin_operation_records_terminal_failure(monkeypatch):
    operation = {'operation_id': 'operation-a', 'status': 'running', 'items': []}
    monkeypatch.setattr(provisioner, '_persistent_container', AsyncMock(side_effect=RuntimeError('runtime gone')))

    await provisioner._activate_pending_plugins('a' * 32, operation)

    assert operation == {
        'operation_id': 'operation-a',
        'status': 'failed',
        'items': [],
        'error': 'runtime gone',
    }


@pytest.mark.asyncio
async def test_existing_runtime_is_reused(monkeypatch):
    scope_id = 'a' * 32
    name = provisioner._runtime_name(scope_id, False)
    existing = {
        'Id': 'container-a',
        'Config': {'Image': 'approved', 'Env': ['API_SERVER_KEY=key']},
        'State': {'Running': True},
        'HostConfig': {'Binds': [f'{name}-plugins:/opt/data/plugins:ro']},
    }
    payload = json.dumps(
        {'scope_id': scope_id, 'broker_url': provisioner.EXPECTED_BROKER_URL, 'ephemeral': False}
    ).encode()
    monkeypatch.setattr(provisioner, '_authenticate', AsyncMock(return_value=payload))
    monkeypatch.setattr(provisioner, '_inspect', AsyncMock(return_value=existing))
    monkeypatch.setattr(provisioner, '_approved_runtime_image', lambda: 'approved')
    monkeypatch.setattr(provisioner, '_verify_gateway_identity', AsyncMock())
    monkeypatch.setattr(provisioner, '_wait_ready', AsyncMock())
    create = AsyncMock()
    monkeypatch.setattr(provisioner, '_create_runtime', create)

    result = await provisioner.ensure_runtime(SimpleNamespace())

    assert result['runtime_id'] == name
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_plugin_validation_failure_rolls_back_and_reaches_terminal_state(monkeypatch):
    scope_id = 'b' * 32
    operation = {'operation_id': 'operation-b', 'status': 'running', 'items': []}
    monkeypatch.setattr(provisioner, '_persistent_container', AsyncMock(return_value={'Id': 'container-b'}))

    async def execute(container_id, command):
        return json.dumps(['safe-plugin--123456abcdef']) if 'root.iterdir()' in ' '.join(command) else ''

    helper_calls = []

    async def helper(runtime_name, command):
        helper_calls.append(command)
        if 'apply' in command:
            raise RuntimeError('enable failed')

    monkeypatch.setattr(provisioner, '_exec', execute)
    monkeypatch.setattr(provisioner, '_run_plugin_helper', helper)
    monkeypatch.setattr(provisioner, '_docker', AsyncMock())
    monkeypatch.setattr(provisioner, '_verify_gateway_identity', AsyncMock())
    monkeypatch.setattr(provisioner, '_wait_ready', AsyncMock())

    await provisioner._activate_pending_plugins(scope_id, operation)

    assert operation['status'] == 'completed'
    assert operation['items'] == [{'name': 'safe-plugin', 'status': 'rolled_back'}]
    assert any('rollback' in command for command in helper_calls)


@pytest.mark.asyncio
async def test_first_plugin_install_snapshots_securely_restarts_and_becomes_ready(monkeypatch):
    scope_id = 'e' * 32
    operation = {'operation_id': 'operation-e', 'status': 'running', 'items': []}
    monkeypatch.setattr(provisioner, '_persistent_container', AsyncMock(return_value={'Id': 'container-e'}))

    async def execute(container_id, command):
        return json.dumps(['safe-plugin--123456abcdef']) if 'root.iterdir()' in ' '.join(command) else ''

    helper_calls = []

    async def helper(runtime_name, command):
        helper_calls.append(command)

    docker = AsyncMock()
    ready = AsyncMock()
    monkeypatch.setattr(provisioner, '_exec', execute)
    monkeypatch.setattr(provisioner, '_run_plugin_helper', helper)
    monkeypatch.setattr(provisioner, '_docker', docker)
    monkeypatch.setattr(provisioner, '_verify_gateway_identity', AsyncMock())
    monkeypatch.setattr(provisioner, '_wait_ready', ready)

    await provisioner._activate_pending_plugins(scope_id, operation)

    assert operation['items'] == [{'name': 'safe-plugin', 'status': 'enabled'}]
    apply_command = next(command for command in helper_calls if 'apply' in command)
    assert 'snapshot.parent.mkdir' in apply_command[1]
    assert 'os.chmod(snapshot.parent,0o700)' in apply_command[1]
    assert any('/restart?t=10' in call.args[1] for call in docker.await_args_list)
    ready.assert_awaited_once_with('container-e')


def test_plugin_supervisor_first_install_and_update_rollback_restore_files_and_config(tmp_path):
    data = tmp_path / 'data'
    plugins = data / 'plugins'
    stage = data / 'plugin-pending' / 'safe-plugin--123456abcdef'
    snapshot = data / 'plugin-snapshots' / 'safe-plugin--123456abcdef'
    bin_dir = tmp_path / 'bin'
    plugins.mkdir(parents=True)
    stage.mkdir(parents=True)
    bin_dir.mkdir()
    (data / 'config.yaml').write_text('enabled: []\n')
    (stage / '__init__.py').write_text('VERSION = 1\n')
    hermes = bin_dir / 'hermes'
    hermes.write_text('#!/bin/sh\nexit 0\n')
    hermes.chmod(0o755)
    script = provisioner.PLUGIN_SUPERVISOR.replace("Path('/data", f"Path('{data}")
    env = {**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}'}

    subprocess.run(
        [sys.executable, '-c', script, 'apply', 'safe-plugin', str(stage), str(snapshot)],
        check=True,
        env=env,
    )
    assert (plugins / 'safe-plugin' / '__init__.py').read_text() == 'VERSION = 1\n'
    assert snapshot.parent.stat().st_mode & 0o777 == 0o700

    (stage / '__init__.py').write_text('VERSION = 2\n')
    (data / 'config.yaml').write_text('enabled: [safe-plugin]\n')
    subprocess.run(
        [sys.executable, '-c', script, 'apply', 'safe-plugin', str(stage), str(snapshot)],
        check=True,
        env=env,
    )
    (data / 'config.yaml').write_text('broken: true\n')
    subprocess.run(
        [sys.executable, '-c', script, 'rollback', 'safe-plugin', str(stage), str(snapshot)],
        check=True,
        env=env,
    )

    assert (plugins / 'safe-plugin' / '__init__.py').read_text() == 'VERSION = 1\n'
    assert (data / 'config.yaml').read_text() == 'enabled: [safe-plugin]\n'


@pytest.mark.asyncio
async def test_memory_seed_uses_version_marker_and_is_idempotent(tmp_path, monkeypatch):
    manifest = approved_manifest()
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(provisioner, 'RELEASE_MANIFEST', str(manifest_path))
    monkeypatch.setattr(provisioner, '_approved_runtime_image', lambda: manifest['runtime_image'])
    image_info = {
        'Config': {
            'User': '10000:10000',
            'Entrypoint': ['/opt/awg/runtime_supervisor.py'],
            'Labels': {
                'org.opencontainers.image.version': manifest['hermes_release'],
                'org.opencontainers.image.revision': manifest['hermes_upstream_commit'],
                'com.awg.hermes.base-image': manifest['upstream_base_image'],
            },
        },
        'RepoDigests': [manifest['runtime_image']],
    }

    writes = []
    created_configs = []

    async def docker(method, path, **kwargs):
        if path.startswith('/images/'):
            return image_info
        if path.startswith('/containers/create'):
            created_configs.append(kwargs['json_body'])
            return {'Id': 'container-memory'}
        return {}

    async def put_files(container_id, root, files):
        writes.append(files)

    monkeypatch.setattr(provisioner, '_docker', docker)
    monkeypatch.setattr(provisioner, '_put_files', put_files)
    monkeypatch.setattr(provisioner, '_verify_gateway_identity', AsyncMock())
    monkeypatch.setattr(provisioner, '_wait_ready', AsyncMock())
    marker_check = AsyncMock(side_effect=RuntimeError('missing marker'))
    monkeypatch.setattr(provisioner, '_exec', marker_check)
    form = provisioner.EnsureRuntimeForm(
        scope_id='c' * 32,
        broker_url=provisioner.EXPECTED_BROKER_URL,
        memory_seed=['concise answers'],
    )

    await provisioner._create_runtime(form)

    assert any(files.get('memories/.openwebui-migration-v1') == b'1\n' for files in writes)
    runtime_env = created_configs[0]['Env']
    assert not any(value.startswith(('OPENAI_API_KEY=', 'AWG_HERMES_QWEN_API_KEY=')) for value in runtime_env)
    assert created_configs[0]['Image'] == manifest['runtime_image']
    writes.clear()
    monkeypatch.setattr(provisioner, '_exec', AsyncMock(return_value=''))
    await provisioner._create_runtime(form)
    assert not any('memories/.openwebui-migration-v1' in files for files in writes)


@pytest.mark.asyncio
async def test_memory_list_delete_and_clear_use_owned_runtime(monkeypatch):
    scope_id = 'd' * 32
    monkeypatch.setattr(
        provisioner,
        '_authenticate',
        AsyncMock(
            side_effect=[
                json.dumps({'scope_id': scope_id}).encode(),
                json.dumps({'scope_id': scope_id, 'name': 'USER.md'}).encode(),
                json.dumps({'scope_id': scope_id}).encode(),
            ]
        ),
    )
    monkeypatch.setattr(provisioner, '_persistent_container', AsyncMock(return_value={'Id': 'container-memory'}))
    execute = AsyncMock(side_effect=['[{"name":"USER.md","content":"concise"}]', '', ''])
    monkeypatch.setattr(provisioner, '_exec', execute)

    assert await provisioner.read_memory(SimpleNamespace()) == {'items': [{'name': 'USER.md', 'content': 'concise'}]}
    assert await provisioner.delete_memory(SimpleNamespace()) == {'status': 'deleted'}
    assert await provisioner.clear_memory(SimpleNamespace()) == {'status': 'cleared'}
