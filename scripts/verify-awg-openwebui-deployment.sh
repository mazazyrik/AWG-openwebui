#!/bin/sh
set -eu

server='root@myai.awg.ru'
compose='/root/Docker/docker-compose.yml'
rag_services='/opt/awg-confluence-rag/deploy/docker-compose.services.yml'
rag_openwebui='/opt/awg-confluence-rag/deploy/docker-compose.openwebui.yml'
rag_ssh_tunnel='/opt/awg-confluence-rag/deploy/docker-compose.ssh-tunnel.yml'
awg_override='/root/Docker/docker-compose.awg-openwebui.yml'
hermes_overlay='/opt/awg-confluence-rag/deploy/docker-compose.hermes.yml'

require_release_target() {
    expected_image=${AWG_OPENWEBUI_IMAGE:-}
    printf '%s\n' "$expected_image" | grep -Eq \
        '^ghcr\.io/mazazyrik/awg-openwebui@sha256:[0-9a-f]{64}$' || {
        echo 'AWG_OPENWEBUI_IMAGE must be the immutable AWG OpenWebUI GHCR digest' >&2
        exit 2
    }
    ssh_port=${AWG_MCP_SSH_PORT:-19101}
    case "$ssh_port" in
        '' | *[!0-9]*)
            echo 'AWG_MCP_SSH_PORT must be a TCP port from 1024 to 65535' >&2
            exit 2
            ;;
    esac
    if [ "$ssh_port" -lt 1024 ] || [ "$ssh_port" -gt 65535 ]; then
        echo 'AWG_MCP_SSH_PORT must be a TCP port from 1024 to 65535' >&2
        exit 2
    fi
}

case "${1:-}" in
    compose)
        require_release_target
        ssh -o BatchMode=yes "$server" "AWG_OPENWEBUI_IMAGE='$expected_image' CONFLUENCE_RAG_MCP_SSH_PORT='$ssh_port' docker compose -f '$compose' -f '$rag_services' -f '$rag_openwebui' -f '$rag_ssh_tunnel' -f '$awg_override' config --quiet && AWG_OPENWEBUI_IMAGE='$expected_image' CONFLUENCE_RAG_MCP_SSH_PORT='$ssh_port' docker compose -f '$compose' -f '$rag_services' -f '$rag_openwebui' -f '$rag_ssh_tunnel' -f '$awg_override' config --format json | jq -e --arg image '$expected_image' --arg port '$ssh_port' '.services.openwebui.image == \$image and .services[\"openwebui-confluence-worker\"].image == \$image and .services.openwebui.environment.CONFLUENCE_RAG_API_URL == \"http://confluence-rag-api:9100\" and .services.openwebui.environment.CONFLUENCE_MCP_URL == \"http://mcp-atlassian:9000/mcp\" and .services.openwebui.environment.CONFLUENCE_QDRANT_URL == \"http://qdrant:6333\" and ((.services[\"mcp-atlassian\"].ports // []) | length == 0) and (.services[\"mcp-atlassian\"].network_mode != \"host\") and (.services[\"confluence-rag-mcp\"].network_mode != \"host\") and (.services.openwebui.networks | has(\"ai\")) and (.services[\"openwebui-confluence-worker\"].networks | has(\"ai\")) and (.services[\"confluence-rag-mcp\"].networks | has(\"ai\")) and ((.services[\"confluence-rag-mcp\"].ports // []) | length == 1) and (.services[\"confluence-rag-mcp\"].ports[0].target == 9101) and ((.services[\"confluence-rag-mcp\"].ports[0].published | tostring) == \$port) and (.services[\"confluence-rag-mcp\"].ports[0].host_ip == \"127.0.0.1\")' >/dev/null && echo compose-ready"
        ;;
    backup)
        ssh -o BatchMode=yes "$server" "test -n \"\$(find /opt/openwebui/backups -maxdepth 1 -type f -name 'webui.db.pre-confluence-*' -print -quit)\" && echo backup-ready"
        ;;
    containers)
        require_release_target
        ssh -o BatchMode=yes "$server" "set -eu; openwebui_id=\$(AWG_OPENWEBUI_IMAGE='$expected_image' CONFLUENCE_RAG_MCP_SSH_PORT='$ssh_port' docker compose -f '$compose' -f '$rag_services' -f '$rag_openwebui' -f '$rag_ssh_tunnel' -f '$awg_override' ps -q openwebui); worker_id=\$(AWG_OPENWEBUI_IMAGE='$expected_image' CONFLUENCE_RAG_MCP_SSH_PORT='$ssh_port' docker compose -f '$compose' -f '$rag_services' -f '$rag_openwebui' -f '$rag_ssh_tunnel' -f '$awg_override' ps -q openwebui-confluence-worker); source_id=\$(AWG_OPENWEBUI_IMAGE='$expected_image' CONFLUENCE_RAG_MCP_SSH_PORT='$ssh_port' docker compose -f '$compose' -f '$rag_services' -f '$rag_openwebui' -f '$rag_ssh_tunnel' -f '$awg_override' ps -q mcp-atlassian); mcp_id=\$(AWG_OPENWEBUI_IMAGE='$expected_image' CONFLUENCE_RAG_MCP_SSH_PORT='$ssh_port' docker compose -f '$compose' -f '$rag_services' -f '$rag_openwebui' -f '$rag_ssh_tunnel' -f '$awg_override' ps -q confluence-rag-mcp); test -n \"\$openwebui_id\"; test -n \"\$worker_id\"; test -n \"\$source_id\"; test -n \"\$mcp_id\"; test \"\$(docker inspect \"\$openwebui_id\" --format '{{.Config.Image}}')\" = '$expected_image'; test \"\$(docker inspect \"\$worker_id\" --format '{{.Config.Image}}')\" = '$expected_image'; test \"\$(docker inspect \"\$openwebui_id\" --format '{{.State.Health.Status}}')\" = healthy; test \"\$(docker inspect \"\$worker_id\" --format '{{.State.Running}}')\" = true; test \"\$(docker inspect \"\$mcp_id\" --format '{{.State.Running}}')\" = true; docker exec \"\$mcp_id\" python -c 'from confluence_rag.mcp_server import mcp; assert mcp.auth.configured'; test \"\$(docker inspect \"\$source_id\" --format '{{.HostConfig.NetworkMode}}')\" != host; test \"\$(docker inspect \"\$mcp_id\" --format '{{.HostConfig.NetworkMode}}')\" != host; test -z \"\$(docker port \"\$source_id\")\"; docker inspect \"\$mcp_id\" --format '{{json .HostConfig.PortBindings}}' | jq -e --arg port '$ssh_port' 'length == 1 and (.\"9101/tcp\" | length == 1) and .\"9101/tcp\"[0].HostIp == \"127.0.0.1\" and .\"9101/tcp\"[0].HostPort == \$port' >/dev/null; echo containers-ready"
        ;;
    network)
        local_port=${AWG_MCP_LOCAL_PORT:-19101}
        token_file=${AWG_MCP_CURSOR_TOKEN_FILE:-}
        case "$local_port" in
            '' | *[!0-9]*)
                echo 'AWG_MCP_LOCAL_PORT must be a TCP port from 1024 to 65535' >&2
                exit 2
                ;;
        esac
        if [ "$local_port" -lt 1024 ] || [ "$local_port" -gt 65535 ]; then
            echo 'AWG_MCP_LOCAL_PORT must be a TCP port from 1024 to 65535' >&2
            exit 2
        fi
        test -f "$token_file" || {
            echo 'AWG_MCP_CURSOR_TOKEN_FILE must name the mode-600 Cursor token file' >&2
            exit 2
        }
        token_mode=$(stat -f %Lp "$token_file" 2>/dev/null || stat -c %a "$token_file")
        test "$token_mode" = 600 || {
            echo 'AWG_MCP_CURSOR_TOKEN_FILE must have mode 600' >&2
            exit 2
        }
        AWG_MCP_CURSOR_TOKEN_FILE="$token_file" python3 - "$local_port" <<'PY'
import http.client
import json
import os
import sys
from pathlib import Path


def payload(body: bytes) -> dict:
    text = body.decode()
    for line in text.splitlines():
        if line.startswith('data:'):
            return json.loads(line.removeprefix('data:').strip())
    return json.loads(text)


port = int(sys.argv[1])
token = Path(os.environ['AWG_MCP_CURSOR_TOKEN_FILE']).read_text().strip()
assert len(token.encode()) >= 32
connection = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
initialize = json.dumps({
    'jsonrpc': '2.0',
    'id': 1,
    'method': 'initialize',
    'params': {
        'protocolVersion': '2025-06-18',
        'capabilities': {},
        'clientInfo': {'name': 'awg-deployment-verifier', 'version': '1'},
    },
})
base_headers = {
    'Accept': 'application/json, text/event-stream',
    'Content-Type': 'application/json',
}
connection.request('POST', '/mcp', body=initialize, headers=base_headers)
response = connection.getresponse()
response.read()
assert response.status == 401
headers = {**base_headers, 'Authorization': f'Bearer {token}'}
connection.request('POST', '/mcp', body=initialize, headers=headers)
response = connection.getresponse()
initialized = payload(response.read())
assert response.status == 200 and 'serverInfo' in initialized.get('result', {})
session_id = response.getheader('mcp-session-id')
assert session_id
session_headers = {**headers, 'Mcp-Session-Id': session_id}
connection.request(
    'POST',
    '/mcp',
    body=json.dumps({'jsonrpc': '2.0', 'method': 'notifications/initialized'}),
    headers=session_headers,
)
response = connection.getresponse()
response.read()
assert response.status in {200, 202}
connection.request(
    'POST',
    '/mcp',
    body=json.dumps({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}}),
    headers=session_headers,
)
response = connection.getresponse()
listed = payload(response.read())
assert response.status == 200
names = {tool['name'] for tool in listed['result']['tools']}
assert names == {'confluence_sync_status', 'search_confluence'}
print('network-ready tools=confluence_sync_status,search_confluence')
PY
        ;;
    module)
        ssh -o BatchMode=yes "$server" "docker exec openwebui python -c \"import sqlite3; import open_webui.integrations.confluence.worker; connection = sqlite3.connect('/app/backend/data/webui.db'); tables = {row[0] for row in connection.execute('select name from sqlite_master where type=\\\"table\\\"')}; assert {'confluence_connection', 'confluence_page', 'confluence_run'} <= tables\" && echo module-ready"
        ;;
    dependencies)
        ssh -o BatchMode=yes "$server" 'docker exec -i openwebui python -' <<'PY'
import asyncio
import os

from qdrant_client import QdrantClient

from open_webui.integrations.confluence.client import ConfluenceAPIClient, ConfluenceMCPClient


async def check():
    api = await ConfluenceAPIClient().check()
    mcp = await ConfluenceMCPClient().check()
    assert api.get('ok') and api.get('enabled')
    assert mcp.get('ok')
    qdrant = QdrantClient(
        url=os.environ['CONFLUENCE_QDRANT_URL'],
        api_key=os.environ.get('CONFLUENCE_QDRANT_API_KEY'),
    )
    qdrant.get_collections()
    qdrant.close()


asyncio.run(check())
PY
        echo dependencies-ready
        ;;
    mcp)
        ssh -o BatchMode=yes "$server" 'docker exec -i openwebui python -' <<'PY'
import asyncio
from types import SimpleNamespace

from open_webui.models.config import Config
from open_webui.models.groups import Groups
from open_webui.models.models import Models
from open_webui.models.users import Users
from open_webui.utils.middleware import connect_mcp_server


def read_grants(grants, *, require_read_only=False):
    result = set()
    for grant in grants or []:
        principal_type = (
            grant.get('principal_type') if isinstance(grant, dict) else grant.principal_type
        )
        principal_id = grant.get('principal_id') if isinstance(grant, dict) else grant.principal_id
        permission = grant.get('permission') if isinstance(grant, dict) else grant.permission
        if permission != 'read':
            assert not require_read_only, 'retrieval MCP grants must be read-only'
            continue
        assert principal_type in {'user', 'group'}, 'retrieval MCP grants must target users or groups'
        assert principal_id and principal_id != '*', 'retrieval MCP grants must not be public'
        result.add((principal_type, principal_id))
    return result


async def check():
    connections = await Config.get('tool_server.connections', []) or []
    matches = [
        connection
        for connection in connections
        if connection.get('type') == 'mcp'
        and connection.get('url') == 'http://confluence-rag-mcp:9101/mcp'
        and connection.get('auth_type') == 'bearer'
        and bool(connection.get('key'))
        and bool((connection.get('config') or {}).get('enable'))
        and bool((connection.get('info') or {}).get('id'))
    ]
    assert len(matches) == 1, 'retrieval MCP connection is not configured'
    connection = matches[0]
    server_id = connection['info']['id']
    tool_id = f'server:mcp:{server_id}'
    connection_grants_raw = connection['config'].get('access_grants') or []
    connection_grants = read_grants(connection_grants_raw, require_read_only=True)
    assert connection_grants, 'retrieval MCP read grants are required'
    assert len(connection_grants) == len(connection_grants_raw), 'retrieval MCP grants are duplicated'

    models = await Models.get_all_models()
    qwen_models = [
        model
        for model in models
        if 'qwen' in ' '.join((model.id, model.base_model_id or '', model.name)).lower()
    ]
    assert qwen_models, 'Qwen model is not configured'
    assigned_qwen_models = [
        model
        for model in qwen_models
        if tool_id in (model.meta.model_dump().get('toolIds') or [])
    ]
    assert assigned_qwen_models, 'retrieval MCP is not assigned to Qwen'
    assert all(
        read_grants(model.access_grants) == connection_grants for model in assigned_qwen_models
    ), 'retrieval MCP grants do not match the assigned Qwen audience'

    target_user = None
    for principal_type, principal_id in sorted(connection_grants):
        user_ids = (
            [principal_id]
            if principal_type == 'user'
            else await Groups.get_group_user_ids_by_id(principal_id)
        )
        for user_id in user_ids:
            user = await Users.get_user_by_id(user_id)
            if user and user.role != 'admin':
                target_user = user
                break
        if target_user:
            break
    assert target_user is not None, 'retrieval MCP grants need a non-admin target user'

    connected = await connect_mcp_server(
        SimpleNamespace(cookies={}),
        server_id,
        target_user,
        {},
        {},
    )
    assert connected is not None, 'non-admin Qwen user cannot access retrieval MCP'
    client, specs = connected
    try:
        names = {spec['name'] for spec in specs or []}
        assert names == {
            'confluence_sync_status',
            'search_confluence',
        }, 'retrieval MCP tools do not match the read-only contract'
        result = await client.call_tool('search_confluence', {'query': 'AWG', 'limit': 1})
        assert result is not None, 'non-admin Qwen retrieval call failed'
    finally:
        await client.disconnect()


asyncio.run(check())
PY
        echo mcp-ready
        ;;
    retrieval)
        ssh -o BatchMode=yes "$server" 'docker exec -i openwebui python -' <<'PY'
import asyncio

from open_webui.integrations.confluence.models import ConfluenceConnections
from open_webui.integrations.confluence.retrieval import retrieve_confluence_knowledge
from open_webui.models.knowledge import Knowledges


async def check():
    connection = await ConfluenceConnections.get_default()
    assert connection and connection.enabled and connection.knowledge_id
    runs = await ConfluenceConnections.list_runs(connection.id, limit=1)
    assert runs and runs[0].status in {'complete', 'incomplete'}
    knowledge = await Knowledges.get_knowledge_by_id(connection.knowledge_id)
    result = await retrieve_confluence_knowledge(knowledge, ['AWG'], 4)
    documents = result['documents'][0]
    metadata = result['metadatas'][0]
    assert documents and len(documents) == len(metadata)
    citation_fields = {'title', 'url', 'page_id', 'version', 'space', 'hash'}
    assert all(citation_fields <= set(item) for item in metadata)


asyncio.run(check())
PY
        echo retrieval-ready
        ;;
    hermes-compose)
        ssh -o BatchMode=yes "$server" "jq -e '.hermes_release == \"v2026.9.14\" and .hermes_upstream_commit == \"345cd2b057a452236de401d3534b8502a7465e8d\" and (.upstream_base_image | type == \"string\" and contains(\"@sha256:\")) and (.runtime_image | type == \"string\" and contains(\"@sha256:\")) and (.verified_at | type == \"string\" and endswith(\"Z\"))' '/opt/awg-confluence-rag/deploy/hermes-release-manifest.json' >/dev/null"
        ssh -o BatchMode=yes "$server" "docker compose -f '$compose' -f '$hermes_overlay' config --quiet && docker compose -f '$compose' -f '$hermes_overlay' config --format json" | jq -e '
            .services["awg-hermes-provisioner"].read_only == true and
            (.services["awg-hermes-provisioner"].cap_drop | index("ALL")) and
            (.services["awg-hermes-provisioner"].security_opt | index("no-new-privileges:true")) and
            ((.services["awg-hermes-provisioner"].ports // []) | length == 0) and
            (.services["awg-hermes-provisioner"].volumes | any(.type == "bind" and .source == "/opt/awg-confluence-rag/deploy/hermes-release-manifest.json" and .target == "/etc/awg-hermes/release-manifest.json" and .read_only == true)) and
            (.services.openwebui.environment.AWG_HERMES_BROKER_URL == "http://openwebui:8080") and
            (.networks["awg-hermes-runtime"].internal == true) and
            (.services.llama.command | join(" ") | contains("-c 131072 -np 2")) and
            (.services.llama.command | join(" ") | contains("q8_0"))
        ' >/dev/null
        echo hermes-compose-ready
        ;;
    hermes-runtime)
        ssh -o BatchMode=yes "$server" 'set -eu; ids=$(docker ps -q --filter label=com.awg.hermes.managed=true); test -n "$ids"; for id in $ids; do docker inspect "$id" --format "{{json .}}" | jq -e '\''
            .Config.User != "" and .Config.User != "0" and
            .Config.Entrypoint == ["/opt/awg/runtime_supervisor.py"] and
            .Config.Cmd == ["gateway","run","--no-supervise"] and
            (.Config.Env | all((startswith("OPENAI_API_KEY=") or startswith("AWG_HERMES_QWEN_API_KEY=")) | not)) and
            (.HostConfig.CapDrop | index("ALL")) and
            .HostConfig.ReadonlyRootfs == true and
            (.HostConfig.SecurityOpt | index("no-new-privileges:true")) and
            .HostConfig.PidsLimit <= 256 and
            (.HostConfig.Binds | all(contains("/opt/data") or contains("/workspace"))) and
            (.HostConfig.Binds | all(contains("docker.sock") | not)) and
            (.NetworkSettings.Ports | to_entries | all(.value == null))
        '\'' >/dev/null; docker top "$id" -n -eo uid,gid,args | awk '\''$1 == 10000 && $2 == 10000 && tolower($0) ~ /gateway/ { ok=1 } END { exit(ok ? 0 : 1) }'\''; done; echo hermes-runtime-ready'
        ;;
    *)
        echo 'usage: verify-awg-openwebui-deployment.sh compose|backup|containers|network|module|dependencies|mcp|retrieval|hermes-compose|hermes-runtime' >&2
        exit 2
        ;;
esac
