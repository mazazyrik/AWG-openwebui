from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import mimetypes
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from uuid import uuid4

import aiohttp
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from open_webui.integrations.confluence.models import ConfluenceConnections
from open_webui.integrations.confluence.retrieval import retrieve_confluence_knowledge
from open_webui.events import EVENTS, publish_event
from open_webui.integrations.hermes.identity import (
    TOKEN_TTL_SECONDS,
    get_authorized_file,
    sign_control_request,
    user_scope_id,
    verify_principal_token,
)
from open_webui.integrations.hermes.policy import authorize_tool
from open_webui.integrations.hermes.settings import (
    HERMES_ALLOWED_ARTIFACT_EXTENSIONS,
    HERMES_ALLOWED_ATTACHMENT_EXTENSIONS,
    HERMES_ARTIFACT_MAX_BYTES,
    HERMES_PROVISIONER_URL,
    HERMES_QWEN_API_KEY,
    HERMES_QWEN_BASE_URL,
    HERMES_QWEN_MODEL,
)
from open_webui.models.chats import Chats
from open_webui.models.files import FileForm, Files
from open_webui.models.knowledge import Knowledges
from open_webui.models.users import Users
from open_webui.storage.provider import Storage
from open_webui.utils.auth import get_verified_user
from open_webui.utils.chat_id import is_saved_chat_id
from open_webui.utils.json_codec import JSONCodec

log = logging.getLogger(__name__)
router = APIRouter()
STAGED_ARTIFACT_INDEX = 'awg:hermes:staged:index'
ARTIFACT_TEXT_MAX_CHARS = 1024 * 1024
ARCHIVE_MAX_ENTRIES = 4096
ARCHIVE_MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
ARCHIVE_MAX_RATIO = 100


class PluginApplyForm(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str = Field(pattern=r'^[a-z][a-z0-9-]{1,63}$')
    files: dict[str, str] = Field(min_length=2, max_length=64)


async def _control_request(method: str, path: str, payload: dict) -> dict:
    body = JSONCodec.dumps(payload, separators=(',', ':')).encode()
    timestamp = str(int(time.time()))
    headers = {
        'Content-Type': 'application/json',
        'X-AWG-Timestamp': timestamp,
        'X-AWG-Signature': sign_control_request(method, path, body, timestamp),
    }
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        async with session.request(method, f'{HERMES_PROVISIONER_URL}{path}', data=body, headers=headers) as response:
            data = await response.json()
            if response.status >= 400:
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='Hermes control failed')
            return data


async def _user_control_with_lease(request: Request, user, path: str, payload: dict) -> dict:
    redis = request.app.state.redis
    if redis is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='Hermes requires Redis')
    lease_key = f'awg:hermes:lease:{user_scope_id(user.id)}'
    lease_id = str(uuid4())
    if not await redis.set(lease_key, lease_id, nx=True, ex=900):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='Hermes user runtime is busy')
    try:
        return await _control_request('POST', path, {'scope_id': user_scope_id(user.id), **payload})
    finally:
        await redis.eval(
            "if redis.call('get',KEYS[1]) == ARGV[1] then return redis.call('del',KEYS[1]) else return 0 end",
            1,
            lease_key,
            lease_id,
        )


async def get_hermes_principal(request: Request, authorization: str = Header(default='')):
    scheme, _, token = authorization.partition(' ')
    if scheme.lower() != 'bearer' or not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Hermes authentication required')
    principal = verify_principal_token(token)
    redis = request.app.state.redis
    if redis is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='Hermes requires Redis')
    active_run = await redis.get(f'awg:hermes:principal:{principal.jti}')
    if isinstance(active_run, bytes):
        active_run = active_run.decode()
    if active_run != principal.run_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Hermes capability is inactive')
    user = await Users.get_user_by_id(principal.user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Hermes user no longer exists')
    return principal, user


@router.get('/health')
async def health():
    return {'status': 'ok'}


@router.post('/tools/{tool_name}')
async def call_tool(tool_name: str, request: Request, principal_context=Depends(get_hermes_principal)):
    principal, user = principal_context
    arguments = await request.json()
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Tool arguments must be an object')
    decision = authorize_tool(principal.scope_id, tool_name, arguments)
    if not decision.allowed or decision.arguments is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f'Tool denied: {decision.reason}')

    if tool_name == 'search_confluence':
        connection = await ConfluenceConnections.get_default()
        knowledge = (
            await Knowledges.get_knowledge_by_id(connection.knowledge_id)
            if connection and connection.knowledge_id
            else None
        )
        if not knowledge:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='Confluence is unavailable')
        result = await retrieve_confluence_knowledge(
            knowledge,
            [decision.arguments.query],
            decision.arguments.limit,
            user=user,
        )
        return {
            'documents': result.get('documents', [[]])[0],
            'sources': result.get('metadatas', [[]])[0],
        }

    if tool_name == 'read_attachment':
        if decision.arguments.file_id not in principal.allowed_file_ids:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='File not found')
        file = await get_authorized_file(decision.arguments.file_id, user)
        extension = Path(file.filename).suffix.lower().lstrip('.')
        if extension not in HERMES_ALLOWED_ATTACHMENT_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail='Unsupported attachment type'
            )
        return {
            'file_id': file.id,
            'filename': file.filename,
            'download_url': f'/api/v1/integrations/hermes/attachments/{file.id}',
        }

    if tool_name == 'stage_plugin':
        if not is_saved_chat_id(principal.chat_id):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Temporary chats cannot persist plugins')
        return await _control_request(
            'POST',
            '/v1/plugins/stage',
            {
                'scope_id': principal.scope_id,
                'name': decision.arguments.name,
                'files': [{'path': path, 'content': content} for path, content in decision.arguments.files.items()],
            },
        )

    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Tool denied: unknown_tool')


@router.get('/attachments/{file_id}')
async def download_attachment(file_id: str, principal_context=Depends(get_hermes_principal)):
    principal, user = principal_context
    if file_id not in principal.allowed_file_ids:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='File not found')
    file = await get_authorized_file(file_id, user)
    extension = Path(file.filename).suffix.lower().lstrip('.')
    if extension not in HERMES_ALLOWED_ATTACHMENT_EXTENSIONS:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail='Unsupported attachment type')
    path = await asyncio.to_thread(Storage.get_file, file.path)
    return FileResponse(path, filename=file.filename, media_type=(file.meta or {}).get('content_type'))


@router.post('/artifacts')
async def upload_artifact(
    request: Request,
    x_awg_filename_b64: str = Header(),
    principal_context=Depends(get_hermes_principal),
):
    principal, user = principal_context
    try:
        filename_bytes = base64.b64decode(x_awg_filename_b64, altchars=b'-_', validate=True)
        filename = os.path.basename(filename_bytes.decode('utf-8').strip())
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid artifact filename')
    if len(filename_bytes) > 255:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Artifact filename is too long')
    extension = Path(filename).suffix.lower().lstrip('.')
    if not filename or extension not in HERMES_ALLOWED_ARTIFACT_EXTENSIONS:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail='Unsupported artifact type')
    content_length = request.headers.get('content-length')
    if content_length and int(content_length) > HERMES_ARTIFACT_MAX_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail='Artifact is too large')
    contents = await request.body()
    if not contents or len(contents) > HERMES_ARTIFACT_MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail='Artifact is empty or too large'
        )
    artifact_id = str(uuid4())
    stored_name = f'hermes-staged-{artifact_id}.{extension}'
    _, file_path = await asyncio.to_thread(
        Storage.upload_file,
        io.BytesIO(contents),
        stored_name,
        {'OpenWebUI-User-Id': user.id, 'AWG-Hermes-Artifact-Id': artifact_id},
    )
    file_hash = hashlib.sha256(contents).hexdigest()
    staged = {
        'artifact_id': artifact_id,
        'filename': filename,
        'path': file_path,
        'hash': file_hash,
        'size': len(contents),
        'content_type': mimetypes.guess_type(filename)[0] or 'application/octet-stream',
        'scope_id': principal.scope_id,
        'user_id': str(user.id),
        'chat_id': principal.chat_id,
        'message_id': principal.message_id,
    }
    artifact_key = f'awg:hermes:artifacts:{principal.jti}'
    staged_json = JSONCodec.dumps(staged, separators=(',', ':'))
    await request.app.state.redis.rpush(artifact_key, staged_json)
    await request.app.state.redis.expire(artifact_key, TOKEN_TTL_SECONDS)
    await request.app.state.redis.zadd(
        STAGED_ARTIFACT_INDEX,
        {staged_json: int(time.time()) + TOKEN_TTL_SECONDS},
    )
    log.info(
        'hermes_artifact_staged actor_scope=%s artifact_id=%s size=%s', principal.scope_id, artifact_id, len(contents)
    )
    return {'artifact_id': artifact_id, 'filename': filename, 'status': 'staged'}


async def _untrack_staged_artifacts(redis, artifacts: list[dict]) -> None:
    if redis is None or not artifacts:
        return
    members = [JSONCodec.dumps(artifact, separators=(',', ':')) for artifact in artifacts]
    await redis.zrem(STAGED_ARTIFACT_INDEX, *members)


async def discard_staged_artifacts(artifacts: list[dict], redis=None) -> None:
    for artifact in artifacts:
        path = artifact.get('path') if isinstance(artifact, dict) else None
        if path:
            await asyncio.to_thread(Storage.delete_file, path)
    await _untrack_staged_artifacts(redis, artifacts)


async def reap_expired_staged_artifacts(redis) -> int:
    members = await redis.zrangebyscore(STAGED_ARTIFACT_INDEX, '-inf', int(time.time()))
    if not members:
        return 0
    artifacts = []
    for member in members:
        if isinstance(member, bytes):
            member = member.decode()
        try:
            artifacts.append(JSONCodec.loads(member))
        except (TypeError, ValueError):
            log.warning('Invalid Hermes staged artifact metadata removed')
    await discard_staged_artifacts(artifacts)
    await redis.zrem(STAGED_ARTIFACT_INDEX, *members)
    return len(artifacts)


async def artifact_reaper_loop(app) -> None:
    while True:
        try:
            await reap_expired_staged_artifacts(app.state.redis)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception('Hermes staged artifact reaper failed')
        await asyncio.sleep(60)


@router.delete('/artifacts/staged')
async def discard_run_artifacts(request: Request, principal_context=Depends(get_hermes_principal)):
    principal, _ = principal_context
    key = f'awg:hermes:artifacts:{principal.jti}'
    async with request.app.state.redis.pipeline(transaction=True) as pipeline:
        pipeline.lrange(key, 0, -1)
        pipeline.delete(key)
        values, _ = await pipeline.execute()
    artifacts = [JSONCodec.loads(value) for value in values]
    await discard_staged_artifacts(artifacts, request.app.state.redis)
    return {'status': 'discarded', 'count': len(artifacts)}


async def commit_staged_artifacts(request: Request, user, artifacts: list[dict]) -> list[dict]:
    committed: list[dict] = []
    try:
        for artifact in artifacts:
            if artifact.get('user_id') != str(user.id):
                raise RuntimeError('Hermes artifact owner mismatch')
            filename = os.path.basename(str(artifact.get('filename') or ''))
            extension = Path(filename).suffix.lower().lstrip('.')
            if extension not in HERMES_ALLOWED_ARTIFACT_EXTENSIONS:
                raise RuntimeError('Hermes artifact type is not approved')
            file_id = str(uuid4())
            file = await Files.insert_new_file(
                user.id,
                FileForm(
                    id=file_id,
                    filename=filename,
                    path=str(artifact['path']),
                    hash=str(artifact['hash']),
                    data={'source': 'hermes'},
                    meta={
                        'name': filename,
                        'content_type': artifact['content_type'],
                        'size': int(artifact['size']),
                        'file_hash': artifact['hash'],
                        'hermes_scope': artifact['scope_id'],
                    },
                ),
            )
            if not file:
                raise RuntimeError('Hermes artifact registration failed')
            payload = {
                'file_id': file.id,
                'id': file.id,
                'type': 'file',
                'filename': file.filename,
                'name': file.filename,
                'content_type': (file.meta or {}).get('content_type'),
                'download_url': f'/api/v1/files/{file.id}/content',
                'url': f'/api/v1/files/{file.id}/content',
            }
            if is_saved_chat_id(artifact['chat_id']) and artifact['message_id']:
                await Chats.insert_chat_files(artifact['chat_id'], artifact['message_id'], [file.id], user.id)
                await Chats.add_message_files_by_id_and_message_id(
                    artifact['chat_id'], artifact['message_id'], [payload]
                )
            await publish_event(
                request, EVENTS.FILE_UPLOADED, actor=user, subject_id=file.id, data={'source': 'hermes'}
            )
            committed.append(payload)
        await _untrack_staged_artifacts(request.app.state.redis, artifacts)
        return committed
    except Exception:
        for payload in committed:
            await Files.delete_file_by_id(payload['id'])
        await discard_staged_artifacts(artifacts, request.app.state.redis)
        raise


def _validate_archive(path: str) -> None:
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if not entries or len(entries) > ARCHIVE_MAX_ENTRIES:
            raise ValueError('Artifact archive entry limit exceeded')
        total = 0
        for entry in entries:
            entry_path = Path(entry.filename)
            if entry.flag_bits & 1 or entry_path.is_absolute() or '..' in entry_path.parts:
                raise ValueError('Artifact archive contains an unsafe entry')
            total += entry.file_size
            if total > ARCHIVE_MAX_UNCOMPRESSED_BYTES:
                raise ValueError('Artifact archive expands beyond the approved limit')
            if entry.file_size > 0 and entry.compress_size == 0:
                raise ValueError('Artifact archive has an invalid compression ratio')
            if entry.compress_size and entry.file_size / entry.compress_size > ARCHIVE_MAX_RATIO:
                raise ValueError('Artifact archive compression ratio is unsafe')


async def extract_staged_artifact_text(artifact: dict, user) -> str:
    del user
    filename = str(artifact['filename'])
    extension = Path(filename).suffix.lower().lstrip('.')
    local_path = await asyncio.to_thread(Storage.get_file, str(artifact['path']))
    if extension in {'docx', 'xlsx', 'pptx'}:
        await asyncio.to_thread(_validate_archive, local_path)
    elif extension == 'pdf':
        magic = await asyncio.to_thread(lambda: Path(local_path).read_bytes()[:5])
        if magic != b'%PDF-':
            raise ValueError('Artifact is not a valid PDF')
    else:
        raise ValueError('Artifact type is not approved')
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        '-m',
        'open_webui.integrations.hermes.artifact_worker',
        local_path,
        extension,
        str(ARTIFACT_TEXT_MAX_CHARS),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=45)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise ValueError('Artifact parsing timed out')
    if process.returncode != 0:
        raise ValueError(f'Artifact parsing failed: {stderr.decode(errors="replace")[:256]}')
    result = json.loads(stdout)
    text = result.get('text') if isinstance(result, dict) else None
    if not isinstance(text, str) or not text.strip() or len(text) > ARTIFACT_TEXT_MAX_CHARS:
        raise ValueError('Artifact text is empty or exceeds the approved limit')
    return text


@router.post('/qwen/v1/chat/completions')
async def qwen_chat_completion(
    request: Request,
    authorization: str = Header(default=''),
    x_awg_run_id: str = Header(default=''),
    x_awg_scope_id: str = Header(default=''),
    x_awg_model: str = Header(default=''),
):
    scheme, _, token = authorization.partition(' ')
    if scheme.lower() != 'bearer' or not token or not HERMES_QWEN_API_KEY or not HERMES_QWEN_MODEL:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Qwen proxy authentication failed')
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    record = await request.app.state.redis.get(f'awg:hermes:qwen:{token_hash}')
    if isinstance(record, bytes):
        record = record.decode()
    try:
        capability = JSONCodec.loads(record)
    except (TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Qwen capability is inactive')
    if capability != {'scope_id': x_awg_scope_id, 'run_id': x_awg_run_id, 'model': x_awg_model}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Qwen capability scope mismatch')
    if x_awg_model != 'awg-qwen':
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Qwen model is not approved')
    payload = await request.json()
    if not isinstance(payload, dict) or payload.get('model') != 'awg-qwen':
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Qwen request model is not approved')
    if any(key in payload for key in ('url', 'base_url', 'api_key')):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Qwen request contains forbidden routing')
    payload['model'] = HERMES_QWEN_MODEL
    headers = {'Authorization': f'Bearer {HERMES_QWEN_API_KEY}', 'Content-Type': 'application/json'}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
        async with session.post(f'{HERMES_QWEN_BASE_URL}/chat/completions', headers=headers, json=payload) as response:
            body = await response.read()
            return Response(content=body, status_code=response.status, media_type=response.content_type)


@router.get('/memory')
async def list_hermes_memory(request: Request, user=Depends(get_verified_user)):
    return await _user_control_with_lease(request, user, '/v1/memory/read', {})


@router.delete('/memory/{name}')
async def delete_hermes_memory(request: Request, name: str, user=Depends(get_verified_user)):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}\.md', name):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid memory name')
    return await _user_control_with_lease(
        request,
        user,
        '/v1/memory/delete',
        {'name': name},
    )


@router.delete('/memory')
async def clear_hermes_memory(request: Request, user=Depends(get_verified_user)):
    return await _user_control_with_lease(request, user, '/v1/memory/clear', {})


@router.post('/plugins/apply')
async def apply_hermes_plugin(request: Request, form: PluginApplyForm, user=Depends(get_verified_user)):
    return await _user_control_with_lease(
        request,
        user,
        '/v1/plugins/stage',
        {
            'name': form.name,
            'files': [{'path': path, 'content': content} for path, content in form.files.items()],
        },
    )
