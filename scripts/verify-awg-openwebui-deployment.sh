#!/bin/sh
set -eu

server='root@myai.awg.ru'
image='ghcr.io/mazazyrik/awg-openwebui:git-205d4b9'
compose='/root/Docker/docker-compose.yml'
rag_services='/opt/awg-confluence-rag/deploy/docker-compose.services.yml'
rag_openwebui='/opt/awg-confluence-rag/deploy/docker-compose.openwebui.yml'
rag_gateway='/opt/awg-confluence-rag/deploy/docker-compose.cursor-mcp.yml'
awg_override='/root/Docker/docker-compose.awg-openwebui.yml'

require_gateway_target() {
    approved_bind_ip=${AWG_MCP_APPROVED_BIND_IP:-}
    approved_vpn_interface=${AWG_MCP_APPROVED_VPN_INTERFACE:-}
    gateway_host=${AWG_MCP_GATEWAY_HOST:-}

    case "$approved_bind_ip" in
        '' | *[!0-9.]*)
            echo 'AWG_MCP_APPROVED_BIND_IP must be an approved private or VPN IPv4 address' >&2
            exit 2
            ;;
    esac
    printf '%s\n' "$approved_bind_ip" | awk -F. '
        NF == 4 && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ &&
        $3 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+$/ &&
        $1 <= 255 && $2 <= 255 && $3 <= 255 && $4 <= 255 &&
        (($1 == 10) || ($1 == 172 && $2 >= 16 && $2 <= 31) ||
         ($1 == 192 && $2 == 168) || ($1 == 100 && $2 >= 64 && $2 <= 127)) { valid = 1 }
        END { exit valid ? 0 : 1 }
    ' || {
        echo 'AWG_MCP_APPROVED_BIND_IP must be an approved private or VPN IPv4 address' >&2
        exit 2
    }
    case "$approved_vpn_interface" in
        '' | *[!A-Za-z0-9_.:-]*)
            echo 'AWG_MCP_APPROVED_VPN_INTERFACE is required' >&2
            exit 2
            ;;
    esac
    case "$gateway_host" in
        '' | localhost | *[!A-Za-z0-9.-]*)
            echo 'AWG_MCP_GATEWAY_HOST is required' >&2
            exit 2
            ;;
    esac
}

case "${1:-}" in
    compose)
        require_gateway_target
        ssh -o BatchMode=yes "$server" "docker compose -f '$compose' -f '$rag_services' -f '$rag_openwebui' -f '$rag_gateway' -f '$awg_override' config --quiet && docker compose -f '$compose' -f '$rag_services' -f '$rag_openwebui' -f '$rag_gateway' -f '$awg_override' config --format json | jq -e --arg bind '$approved_bind_ip' '.services.openwebui.image == \"$image\" and .services[\"openwebui-confluence-worker\"].image == \"$image\" and .services.openwebui.environment.CONFLUENCE_RAG_API_URL == \"http://confluence-rag-api:9100\" and .services.openwebui.environment.CONFLUENCE_MCP_URL == \"http://mcp-atlassian:9000/mcp\" and .services.openwebui.environment.CONFLUENCE_QDRANT_URL == \"http://qdrant:6333\" and ((.services[\"mcp-atlassian\"].ports // []) | length == 0) and ((.services[\"confluence-rag-mcp\"].ports // []) | length == 0) and (.services[\"mcp-atlassian\"].network_mode != \"host\") and (.services[\"confluence-rag-mcp\"].network_mode != \"host\") and (.services.openwebui.networks | has(\"ai\")) and (.services[\"openwebui-confluence-worker\"].networks | has(\"ai\")) and (.services[\"confluence-rag-mcp\"].networks | has(\"ai\")) and ((.services[\"confluence-rag-gateway\"].ports // []) | length == 1) and (.services[\"confluence-rag-gateway\"].ports[0].target == 443) and ((.services[\"confluence-rag-gateway\"].ports[0].published | tostring) == \"443\") and (.services[\"confluence-rag-gateway\"].ports[0].host_ip == \$bind)' >/dev/null && echo compose-ready"
        ;;
    backup)
        ssh -o BatchMode=yes "$server" "test -n \"\$(find /opt/openwebui/backups -maxdepth 1 -type f -name 'webui.db.pre-confluence-*' -print -quit)\" && echo backup-ready"
        ;;
    containers)
        require_gateway_target
        ssh -o BatchMode=yes "$server" "set -eu; openwebui_id=\$(docker compose -f '$compose' -f '$rag_services' -f '$rag_openwebui' -f '$rag_gateway' -f '$awg_override' ps -q openwebui); worker_id=\$(docker compose -f '$compose' -f '$rag_services' -f '$rag_openwebui' -f '$rag_gateway' -f '$awg_override' ps -q openwebui-confluence-worker); source_id=\$(docker compose -f '$compose' -f '$rag_services' -f '$rag_openwebui' -f '$rag_gateway' -f '$awg_override' ps -q mcp-atlassian); mcp_id=\$(docker compose -f '$compose' -f '$rag_services' -f '$rag_openwebui' -f '$rag_gateway' -f '$awg_override' ps -q confluence-rag-mcp); gateway_id=\$(docker compose -f '$compose' -f '$rag_services' -f '$rag_openwebui' -f '$rag_gateway' -f '$awg_override' ps -q confluence-rag-gateway); test -n \"\$openwebui_id\"; test -n \"\$worker_id\"; test -n \"\$source_id\"; test -n \"\$mcp_id\"; test -n \"\$gateway_id\"; test \"\$(docker inspect \"\$openwebui_id\" --format '{{.Config.Image}}')\" = '$image'; test \"\$(docker inspect \"\$worker_id\" --format '{{.Config.Image}}')\" = '$image'; test \"\$(docker inspect \"\$openwebui_id\" --format '{{.State.Health.Status}}')\" = healthy; test \"\$(docker inspect \"\$worker_id\" --format '{{.State.Running}}')\" = true; test \"\$(docker inspect \"\$mcp_id\" --format '{{.State.Running}}')\" = true; test \"\$(docker inspect \"\$gateway_id\" --format '{{.State.Running}}')\" = true; docker exec \"\$mcp_id\" python -c 'from confluence_rag.mcp_server import mcp; assert mcp.auth.configured'; test \"\$(docker inspect \"\$source_id\" --format '{{.HostConfig.NetworkMode}}')\" != host; test \"\$(docker inspect \"\$mcp_id\" --format '{{.HostConfig.NetworkMode}}')\" != host; test -z \"\$(docker port \"\$source_id\")\"; test -z \"\$(docker port \"\$mcp_id\")\"; docker inspect \"\$gateway_id\" --format '{{json .HostConfig.PortBindings}}' | jq -e --arg bind '$approved_bind_ip' 'length == 1 and (.\"443/tcp\" | length == 1) and .\"443/tcp\"[0].HostIp == \$bind and .\"443/tcp\"[0].HostPort == \"443\"' >/dev/null; ip -o -4 addr show dev '$approved_vpn_interface' | awk '{print \$4}' | cut -d/ -f1 | grep -Fqx '$approved_bind_ip'; echo containers-ready"
        ;;
    network)
        require_gateway_target
        auth_status=$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 10 --resolve "$gateway_host:443:$approved_bind_ip" "https://$gateway_host/mcp")
        args_status=$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 10 --resolve "$gateway_host:443:$approved_bind_ip" "https://$gateway_host/mcp?probe=1")
        test "$auth_status" = 401
        test "$args_status" = 404
        echo network-ready
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
    *)
        echo 'usage: verify-awg-openwebui-deployment.sh compose|backup|containers|network|module|dependencies|mcp|retrieval' >&2
        exit 2
        ;;
esac
