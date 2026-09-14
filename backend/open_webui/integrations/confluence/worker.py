import asyncio
import logging
import os
import socket
import time
import uuid
from typing import Any

from open_webui.integrations.confluence.client import (
    ConfluenceAPIClient,
    ConfluenceClientError,
    ConfluenceMCPClient,
    secret_presence,
)
from open_webui.integrations.confluence.index import indexed_pages
from open_webui.integrations.confluence.models import ConfluenceConnections

log = logging.getLogger(__name__)

TERMINAL_STATUSES = {'complete', 'incomplete', 'failed'}


async def _wait_for_remote_run(
    client: ConfluenceAPIClient,
    remote_run_id: str,
    local_run_id: str,
    owner_id: str,
) -> dict[str, Any]:
    timeout = max(60, int(os.getenv('CONFLUENCE_SYNC_TIMEOUT_SECONDS', '3600')))
    poll_interval = max(1, int(os.getenv('CONFLUENCE_SYNC_POLL_SECONDS', '5')))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not await ConfluenceConnections.heartbeat_run(local_run_id, owner_id):
            raise ConfluenceClientError('worker_lease_lost')
        remote = await client.status()
        if remote and str(remote.get('run_id')) == remote_run_id:
            if remote.get('status') in TERMINAL_STATUSES:
                return remote
        await asyncio.sleep(poll_interval)
    raise ConfluenceClientError('sync_timeout')


def _remote_counters(remote: dict[str, Any]) -> dict[str, int]:
    names = (
        'pages_seen',
        'pages_indexed',
        'pages_skipped',
        'pages_deleted',
        'pages_failed',
    )
    return {name: int(remote.get(name) or 0) for name in names}


async def run_worker_once() -> None:
    connection = await ConfluenceConnections.get_default()
    if not connection or not connection.enabled:
        log.info('confluence_worker_idle reason=disabled')
        return

    owner_id = f'{socket.gethostname()}-{uuid.uuid4()}'
    run = await ConfluenceConnections.acquire_next_run(owner_id)
    if not run:
        log.info(
            'confluence_worker_idle reason=no_queued_runs connection_id=%s',
            connection.id,
        )
        return

    log.info(
        'confluence_worker_run_started run_id=%s mode=%s connection_id=%s',
        run.id,
        run.mode,
        connection.id,
    )
    try:
        client = ConfluenceAPIClient()
        remote = await client.trigger_sync(run.mode)
        remote = await _wait_for_remote_run(
            client,
            str(remote['run_id']),
            run.id,
            owner_id,
        )
        remote_status = str(remote.get('status'))
        counters = _remote_counters(remote)
        if remote_status in {'complete', 'incomplete'}:
            pages = await asyncio.to_thread(indexed_pages, connection)
            counters.update(await ConfluenceConnections.sync_pages(connection.id, pages))
        safe_error_code = str(remote.get('error_code') or '') or None
        await ConfluenceConnections.finish_run(
            run.id,
            owner_id,
            remote_status,
            counters=counters,
            safe_error_code=safe_error_code,
        )
        await ConfluenceConnections.update_state(
            connection.id,
            remote_status,
            watermark=str(remote.get('watermark') or '') or None,
            last_error_code=safe_error_code,
        )
    except Exception as error:
        safe_error_code = (str(error) if isinstance(error, ConfluenceClientError) else type(error).__name__)[:100]
        await ConfluenceConnections.finish_run(
            run.id,
            owner_id,
            'failed',
            safe_error_code=safe_error_code,
        )
        await ConfluenceConnections.update_state(
            connection.id,
            'failed',
            last_error_code=safe_error_code,
        )
        log.exception(
            'confluence_worker_run_failed run_id=%s error_code=%s',
            run.id,
            safe_error_code,
        )


async def run_worker(poll_interval: int = 30) -> None:
    while True:
        await run_worker_once()
        await asyncio.sleep(poll_interval)


async def run_import(mode: str = 'full', dry_run: bool = True) -> dict[str, Any]:
    connection = await ConfluenceConnections.get_default()
    if dry_run:
        return {
            'dry_run': True,
            'connection_enabled': bool(connection and connection.enabled),
            'api': await ConfluenceAPIClient().check(),
            'mcp': await ConfluenceMCPClient().check(),
            'secrets': secret_presence(),
        }
    if not connection or not connection.enabled:
        raise ConfluenceClientError('connection_disabled')

    run = await ConfluenceConnections.queue_run(connection.id, mode)
    while run.status in {'queued', 'running'}:
        await run_worker_once()
        current = await ConfluenceConnections.get_run(run.id)
        if current is None:
            raise ConfluenceClientError('local_run_missing')
        run = current
        if run.status in {'queued', 'running'}:
            await asyncio.sleep(1)
    return {'dry_run': False, 'run': run.model_dump()}
