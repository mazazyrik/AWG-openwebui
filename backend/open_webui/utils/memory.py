from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Any

from fastapi import HTTPException
from open_webui.config import RAG_EMBEDDING_CONTENT_PREFIX
from open_webui.integrations.confluence.runtime import get_awg_request_state
from open_webui.integrations.confluence.scope_router import MemoryCommand
from open_webui.models.config import Config
from open_webui.models.memories import Memories, MemoryModel
from open_webui.models.users import UserModel
from open_webui.retrieval.vector.async_client import ASYNC_VECTOR_DB_CLIENT
from open_webui.utils.access_control import has_permission
from open_webui.utils.json_codec import JSONCodec
from open_webui.utils.misc import add_or_update_system_message, get_content_from_message

log = logging.getLogger(__name__)

MEMORY_CONTEXT_OPEN = '<memory_context>'
MEMORY_CONTEXT_CLOSE = '</memory_context>'
AWG_PREFERENCES_OPEN = '<personal_preferences source="awg_gpt_memory">'
AWG_PREFERENCES_CLOSE = '</personal_preferences>'
AWG_MEMORY_PATH = 'awg-gpt'
AWG_MEMORY_COLLECTION_PREFIX = 'user-memory-awg-gpt'
AWG_MEMORY_MAX_VALUE_CHARS = 300
AWG_MEMORY_SECRET_RE = re.compile(
    r'\b(?:password|парол\w*|secret|секрет\w*|api[-_ ]?key|токен\w*|bearer|private[-_ ]?key)\b|'
    r'\bsk-[A-Za-z0-9_-]{12,}\b|\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\.|-----BEGIN [A-Z ]+PRIVATE KEY-----',
    re.IGNORECASE,
)
AWG_MEMORY_POLICY_RE = re.compile(
    r'\b(?:ignore|override|disregard).{0,40}(?:instruction|rule|prompt)|'
    r'\b(?:игнорируй|переопредели|забудь).{0,40}(?:инструкц\w*|правил\w*|промпт)|'
    r'\b(?:выдай|дай|назначь|сделай)\b.{0,40}'
    r'\b(?:админ\w*|администратор\w*|роль\w*|прав\w*|доступ\w*|разрешени\w*)\b|'
    r'\b(?:роль\w*|прав\w*|доступ\w*|разрешени\w*)\b.{0,30}'
    r'\b(?:админ\w*|администратор\w*)\b|'
    r'\b(?:grant|give|make|assign)\b.{0,40}'
    r'\b(?:admin(?:istrator)?|role|permissions?|access)\b|'
    r'\b(?:admin(?:istrator)?|system)\s+(?:role|permissions?|access|prompt)\b|'
    r'\b(?:SOURCE_DATA_JSON|FINAL_ROUTE|AWG_GPT_POLICY)\b|<\/?(?:system|assistant|user)>|https?://|\bwww\.',
    re.IGNORECASE,
)
AWG_MEMORY_CORPORATE_RE = re.compile(
    r'\b(?:директор|ceo|руководитель|сотрудник|штат|оборот|выручка|прибыль|владелец|employee|headcount|revenue)'
    r'\b.{0,80}\bAWG\b|'
    r'\bAWG\b.{0,80}\b(?:директор|ceo|руководитель|сотрудник|штат|оборот|выручка|прибыль|владелец|employee|'
    r'headcount|revenue|основан\w*|founded)\b|\bAWG\b\s*(?:—\s*|(?:это|является|is|was|has|имеет)\b)',
    re.IGNORECASE,
)


def clean_memory_content(content: str | None) -> str:
    value = (content or '').strip()
    if not value:
        raise HTTPException(status_code=400, detail='Memory content cannot be empty')
    return value


def clean_memory_path(path: str | None) -> str | None:
    value = re.sub(r'/+', '/', (path or '').strip().strip('/'))
    if not value:
        return None
    parts = value.split('/')
    if any(part in {'', '.', '..'} for part in parts) or any(ord(char) < 32 for char in value):
        raise HTTPException(status_code=400, detail='Invalid memory path')
    return value


def memory_vector_text(content: str, path: str | None = None) -> str:
    path = clean_memory_path(path)
    return f'{path}\n{content}' if path else content


def memory_label(memory) -> str:
    return f'{memory.path}: {memory.content}' if memory.path else memory.content


async def _check_awg_memory_permission(user: UserModel) -> None:
    config = await Config.get_many('memories.enable', 'user.permissions')
    if not config.get('memories.enable'):
        raise HTTPException(status_code=404, detail='Memory is disabled')
    if user.role != 'admin' and not await has_permission(user.id, 'features.memories', config.get('user.permissions')):
        raise HTTPException(status_code=403, detail='Memory access denied')


def _validate_awg_memory_value(value: str | None, *, allow_awg_fact: bool = False) -> str:
    text = clean_memory_content(value)
    normalized = re.sub(r'\s+', ' ', text).casefold()
    if len(text) > AWG_MEMORY_MAX_VALUE_CHARS:
        raise HTTPException(status_code=400, detail='Memory value is too long')
    if AWG_MEMORY_SECRET_RE.search(normalized) or AWG_MEMORY_POLICY_RE.search(normalized):
        raise HTTPException(status_code=400, detail='Memory value is not allowed')
    if not allow_awg_fact and AWG_MEMORY_CORPORATE_RE.search(text):
        raise HTTPException(status_code=400, detail='Corporate facts cannot be stored as personal memory')
    return text


def _awg_memory_metadata(memory: MemoryModel) -> dict:
    return {
        'created_at': memory.created_at,
        'updated_at': memory.updated_at,
        'type': memory.type,
        'path': memory.path,
    }


async def _awg_memory_vector(request, user: UserModel, memory: MemoryModel, vector: list) -> None:
    await ASYNC_VECTOR_DB_CLIENT.upsert(
        collection_name=f'{AWG_MEMORY_COLLECTION_PREFIX}-{user.id}',
        items=[
            {
                'id': memory.id,
                'text': memory_vector_text(memory.content, memory.path),
                'vector': vector,
                'metadata': _awg_memory_metadata(memory),
            }
        ],
    )


async def _embed_awg_memory(request, user: UserModel, content: str, path: str) -> list:
    return await request.app.state.EMBEDDING_FUNCTION(
        memory_vector_text(content, path),
        prefix=RAG_EMBEDDING_CONTENT_PREFIX,
        user=user,
    )


def _awg_memory_rows(memories: list[MemoryModel]) -> list[MemoryModel]:
    return [
        memory
        for memory in memories
        if memory.type == 'user'
        and isinstance(memory.meta, dict)
        and memory.meta.get('created_by') == 'awg_gpt_explicit'
        and memory.meta.get('kind') in {'alias', 'preference'}
        and isinstance(memory.path, str)
        and memory.path.startswith(f'{AWG_MEMORY_PATH}/')
    ]


async def _delete_awg_memory(request, user: UserModel, memories: list[MemoryModel], command: MemoryCommand) -> dict:
    lookup = _validate_awg_memory_value(command.value, allow_awg_fact=True).casefold()
    selected = next(
        (
            memory
            for memory in memories
            if lookup
            in {
                str(memory.meta.get('alias') or '').casefold(),
                str(memory.meta.get('value') or '').casefold(),
                memory.content.casefold(),
            }
        ),
        None,
    )
    if selected is None:
        raise HTTPException(status_code=404, detail='Memory entry not found')
    old_vector = await _embed_awg_memory(request, user, selected.content, selected.path or AWG_MEMORY_PATH)
    await ASYNC_VECTOR_DB_CLIENT.delete(
        collection_name=f'{AWG_MEMORY_COLLECTION_PREFIX}-{user.id}',
        ids=[selected.id],
    )
    deleted = await Memories.delete_memory_by_id_and_user_id(
        selected.id,
        user.id,
        include_awg_gpt=True,
    )
    if not deleted:
        await _awg_memory_vector(request, user, selected, old_vector)
        raise HTTPException(status_code=409, detail='Memory delete failed')
    return {'status': 'deleted', 'kind': selected.meta.get('kind')}


async def _save_awg_memory(
    request,
    user: UserModel,
    memories: list[MemoryModel],
    command: MemoryCommand,
    reserved_aliases: tuple[str, ...],
) -> dict:
    if command.kind not in {'alias', 'preference'} or not command.value:
        raise HTTPException(status_code=400, detail='Memory command is incomplete')

    value = _validate_awg_memory_value(command.value)
    alias = _validate_awg_memory_value(command.key) if command.kind == 'alias' else None
    reserved = {'awg', 'avg', 'авг', 'awg gpt', 'avg gpt', 'авг gpt'} | {item.casefold() for item in reserved_aliases}
    if alias and alias.casefold() in reserved:
        raise HTTPException(status_code=400, detail='Reserved aliases cannot be changed')
    fingerprint = hashlib.sha256((alias or value).casefold().encode('utf-8')).hexdigest()[:16]
    path = f'{AWG_MEMORY_PATH}/{command.kind}/{fingerprint}'
    content = (
        f'AWG project alias: {alias} means {value}'
        if command.kind == 'alias'
        else f'AWG GPT response preference: {value}'
    )
    meta = {
        'created_by': 'awg_gpt_explicit',
        'kind': command.kind,
        'alias': alias,
        'value': value,
    }
    existing = next(
        (
            memory
            for memory in memories
            if memory.meta.get('kind') == command.kind
            and (
                (command.kind == 'preference' and memory.meta.get('value', '').casefold() == value.casefold())
                or (command.kind == 'alias' and memory.meta.get('alias', '').casefold() == (alias or '').casefold())
            )
        ),
        None,
    )
    vector = await _embed_awg_memory(request, user, content, path)
    if existing is None:
        results = await Memories.apply_memory_operations(
            user.id,
            [{'action': 'add', 'type': 'user', 'path': path, 'content': content, 'meta': meta}],
            include_awg_gpt=True,
        )
        memory = results[0].get('memory') if results else None
        if not isinstance(memory, MemoryModel):
            raise HTTPException(status_code=409, detail='Memory create failed')
        if results[0].get('status') == 'skipped' and (
            not isinstance(memory.meta, dict) or memory.meta.get('created_by') != 'awg_gpt_explicit'
        ):
            raise HTTPException(status_code=409, detail='Memory path is already in use')
        try:
            await _awg_memory_vector(request, user, memory, vector)
        except Exception:
            await Memories.delete_memory_by_id_and_user_id(
                memory.id,
                user.id,
                include_awg_gpt=True,
            )
            raise
        return {'status': results[0].get('status'), 'kind': command.kind}

    old_content = existing.content
    old_path = existing.path or AWG_MEMORY_PATH
    old_meta = dict(existing.meta or {})
    old_vector = await _embed_awg_memory(request, user, old_content, old_path)
    results = await Memories.apply_memory_operations(
        user.id,
        [
            {
                'action': 'replace',
                'id': existing.id,
                'type': 'user',
                'path': path,
                'content': content,
                'meta': meta,
            }
        ],
        include_awg_gpt=True,
    )
    memory = results[0].get('memory') if results else None
    if not isinstance(memory, MemoryModel):
        raise HTTPException(status_code=409, detail='Memory update failed')
    try:
        await _awg_memory_vector(request, user, memory, vector)
    except Exception:
        restored = await Memories.apply_memory_operations(
            user.id,
            [
                {
                    'action': 'replace',
                    'id': existing.id,
                    'type': existing.type,
                    'path': old_path,
                    'content': old_content,
                    'meta': old_meta,
                }
            ],
            include_awg_gpt=True,
        )
        restored_memory = restored[0].get('memory') if restored else None
        if isinstance(restored_memory, MemoryModel):
            await _awg_memory_vector(request, user, restored_memory, old_vector)
        raise
    return {'status': 'updated', 'kind': command.kind}


async def execute_awg_memory_command(
    request,
    user_data: dict,
    command: MemoryCommand,
    reserved_aliases: tuple[str, ...] = (),
) -> dict:
    """Apply an explicit AWG alias or preference command for the authenticated user."""
    user = UserModel(**user_data)
    await _check_awg_memory_permission(user)
    memories = _awg_memory_rows(await Memories.get_memories_by_user_id(user.id, include_awg_gpt=True) or [])
    if command.operation == 'list':
        return {
            'status': 'read',
            'aliases': len([memory for memory in memories if memory.meta.get('kind') == 'alias']),
            'preferences': len([memory for memory in memories if memory.meta.get('kind') == 'preference']),
        }
    if command.operation == 'remove':
        return await _delete_awg_memory(request, user, memories, command)
    return await _save_awg_memory(request, user, memories, command, reserved_aliases)


async def get_awg_alias_expansions(request, user_data: dict, question: str) -> list[str]:
    """Return validated per-user alias targets found in the current question."""
    user = UserModel(**user_data)
    await _check_awg_memory_permission(user)
    matches = []
    for memory in _awg_memory_rows(await Memories.get_memories_by_user_id(user.id, include_awg_gpt=True) or []):
        if memory.meta.get('kind') != 'alias':
            continue
        alias = _validate_awg_memory_value(memory.meta.get('alias'), allow_awg_fact=True)
        value = _validate_awg_memory_value(memory.meta.get('value'))
        if re.search(rf'(?<!\w){re.escape(alias)}(?!\w)', question, re.IGNORECASE):
            matches.append(value)
        if len(matches) == 3:
            break
    return matches


def _path_parts(path: str | None) -> list[str]:
    return [part for part in (path or '').split('/') if part]


def _parent_path(path: str | None) -> str | None:
    parts = _path_parts(path)
    return '/'.join(parts[:-1]) if len(parts) > 1 else None


def _path_rank(memory_path: str | None, lookup_path: str | None) -> tuple | None:
    if not lookup_path:
        return None

    memory_path = clean_memory_path(memory_path)
    lookup_path = clean_memory_path(lookup_path)
    if not memory_path or not lookup_path:
        return None

    if memory_path == lookup_path:
        return (0, 0)
    if memory_path.startswith(f'{lookup_path}/'):
        return (1, len(_path_parts(memory_path)) - len(_path_parts(lookup_path)))
    if lookup_path.startswith(f'{memory_path}/'):
        return (2, len(_path_parts(lookup_path)) - len(_path_parts(memory_path)))
    if _parent_path(memory_path) and _parent_path(memory_path) == _parent_path(lookup_path):
        return (3, 0)

    memory_parts = set(_path_parts(memory_path))
    lookup_parts = set(_path_parts(lookup_path))
    shared = len(memory_parts & lookup_parts)
    if shared:
        return (4, -shared)
    if _path_parts(memory_path)[-1:] == _path_parts(lookup_path)[-1:]:
        return (5, 0)

    return None


def _memory_matches_query(memory, query: str) -> bool:
    value = query.strip().lower()
    if not value:
        return True
    return value in (memory.content or '').lower() or value in (memory.path or '').lower()


def search_memory_rows(
    memories: list,
    *,
    query: str | None = None,
    path: str | None = None,
    memory_id: str | None = None,
    memory_type: str = 'all',
    limit: int = 20,
) -> list:
    rows = list(memories or [])
    if memory_id:
        rows = [memory for memory in rows if memory.id == memory_id]
    if memory_type != 'all':
        rows = [memory for memory in rows if memory.type == memory_type]

    query = (query or '').strip()
    lookup_path = clean_memory_path(path)
    if lookup_path:
        basename = _path_parts(lookup_path)[-1] if _path_parts(lookup_path) else lookup_path

        def related(memory) -> bool:
            rank = _path_rank(memory.path, lookup_path)
            if rank is not None:
                return True
            haystack = f'{memory.path or ""}\n{memory.content or ""}'.lower()
            return lookup_path.lower() in haystack or basename.lower() in haystack

        rows = [memory for memory in rows if related(memory)]

    if query:
        rows = [memory for memory in rows if _memory_matches_query(memory, query)]

    def sort_key(memory):
        rank = _path_rank(memory.path, lookup_path) if lookup_path else None
        return rank if rank is not None else (9, 0), -(memory.updated_at or 0), memory.id or ''

    return sorted(rows, key=sort_key)[: max(1, min(limit or 20, 100))]


def list_memory_path_groups(
    memories: list,
    *,
    query: str = '',
    memory_type: str = 'all',
    limit: int = 100,
) -> dict:
    rows = [
        memory
        for memory in (memories or [])
        if (memory_type == 'all' or memory.type == memory_type) and _memory_matches_query(memory, query)
    ]
    grouped: dict[tuple[str | None, str], dict] = {}
    for memory in rows:
        key = (memory.path, memory.type)
        group = grouped.setdefault(
            key,
            {
                'path': memory.path,
                'type': memory.type,
                'count': 0,
                'updated_at': 0,
                'children': [],
            },
        )
        group['count'] += 1
        group['updated_at'] = max(group['updated_at'], memory.updated_at or 0)

    paths = [path for path, _ in grouped if path]
    for group in grouped.values():
        path = group['path']
        if not path:
            continue
        prefix = f'{path}/'
        children = []
        for candidate in paths:
            if not candidate.startswith(prefix):
                continue
            remainder = candidate[len(prefix) :]
            child = f'{prefix}{remainder.split("/", 1)[0]}'
            if child not in children:
                children.append(child)
        group['children'] = children[:20]

    groups = sorted(grouped.values(), key=lambda item: item['updated_at'], reverse=True)
    return {'paths': groups[: max(1, min(limit or 100, 500))], 'count': len(groups)}


def read_memory_path_rows(
    memories: list,
    *,
    path: str,
    memory_type: str = 'all',
    include_children: bool = True,
    limit: int = 50,
) -> dict:
    lookup_path = clean_memory_path(path)
    if not lookup_path:
        raise HTTPException(status_code=400, detail='Memory path is required')

    rows = [memory for memory in (memories or []) if memory_type == 'all' or memory.type == memory_type]
    path_set = {memory.path for memory in rows if memory.path}
    parents = [
        '/'.join(_path_parts(lookup_path)[:idx])
        for idx in range(1, len(_path_parts(lookup_path)))
        if '/'.join(_path_parts(lookup_path)[:idx]) in path_set
    ]
    children = sorted(
        {
            f'{lookup_path}/{memory.path[len(lookup_path) + 1 :].split("/", 1)[0]}'
            for memory in rows
            if memory.path and memory.path.startswith(f'{lookup_path}/')
        }
    )

    def selected(memory) -> bool:
        if memory.path == lookup_path:
            return True
        if memory.path in parents:
            return True
        return bool(include_children and memory.path and memory.path.startswith(f'{lookup_path}/'))

    selected_rows = [memory for memory in rows if selected(memory)]

    def sort_key(memory):
        if memory.path == lookup_path:
            return (0, 0, -(memory.updated_at or 0), memory.id or '')
        if memory.path and memory.path.startswith(f'{lookup_path}/'):
            return (1, len(_path_parts(memory.path)), -(memory.updated_at or 0), memory.id or '')
        return (2, -len(_path_parts(memory.path)), -(memory.updated_at or 0), memory.id or '')

    return {
        'path': lookup_path,
        'parents': parents,
        'children': children[:50],
        'memories': sorted(selected_rows, key=sort_key)[: max(1, min(limit or 50, 100))],
    }


def memory_path_hints(query: str, memories: list, limit: int = 6) -> list[str]:
    lowered = (query or '').lower()
    if not lowered:
        return []

    hints: list[str] = []
    for memory in sorted(memories or [], key=lambda item: (item.path or '', item.content or '', item.id or '')):
        path = memory.path
        if not path or path in hints:
            continue
        parts = _path_parts(path)
        last = parts[-1] if parts else path
        if path.lower() in lowered or last.lower() in lowered:
            hints.append(path)
        elif any(len(part) >= 3 and part.lower() in lowered for part in parts):
            hints.append(path)
        if len(hints) >= limit:
            break
    return hints


def validate_memory_operations(form_data) -> list[dict]:
    if not form_data.operations:
        raise HTTPException(status_code=400, detail='No memory operations provided')

    operations = []
    for operation in form_data.operations:
        op = operation.model_dump()
        action = op.get('action')

        if action == 'add':
            op['content'] = clean_memory_content(op.get('content'))
            op['type'] = Memories.normalize_memory_type(op.get('type'))
            op['path'] = clean_memory_path(op.get('path'))
        elif action == 'replace':
            if not op.get('id'):
                raise HTTPException(status_code=400, detail='Memory id is required for replace')
            op['content'] = clean_memory_content(op.get('content'))
            if op.get('type') is not None:
                op['type'] = Memories.normalize_memory_type(op.get('type'))
            op['path'] = clean_memory_path(op.get('path'))
        elif action == 'move':
            if not op.get('id'):
                raise HTTPException(status_code=400, detail='Memory id is required for move')
            op['path'] = clean_memory_path(op.get('path'))
        elif action == 'remove':
            if not op.get('id'):
                raise HTTPException(status_code=400, detail='Memory id is required for remove')
        else:
            raise HTTPException(status_code=400, detail=f'Unsupported memory operation: {action}')

        operations.append(op)

    return operations


def model_allows_memory(model: dict | None) -> bool:
    return ((model or {}).get('info', {}).get('meta', {}).get('capabilities') or {}).get('memory', True)


async def _add_awg_preference_context(form_data: dict, user) -> dict:
    memories = _awg_memory_rows(await Memories.get_memories_by_user_id(user.id, include_awg_gpt=True) or [])
    preferences = []
    for memory in memories:
        if memory.meta.get('kind') != 'preference':
            continue
        try:
            preferences.append(_validate_awg_memory_value(memory.meta.get('value')))
        except HTTPException:
            continue
    if not preferences:
        return form_data

    config = await Config.get_many('memories.user_char_limit')
    try:
        limit = max(250, int(config.get('memories.user_char_limit') or 2000))
    except Exception:
        limit = 2000
    rendered = '\n'.join(f'- {value}' for value in preferences)[:limit]
    context = f'{AWG_PREFERENCES_OPEN}\n{rendered}\n{AWG_PREFERENCES_CLOSE}'
    form_data['messages'] = add_or_update_system_message(context, form_data['messages'], append=True)
    return form_data


async def add_memory_context(request, form_data: dict, user, model: dict | None = None):
    is_awg_request, awg_state = get_awg_request_state(
        request,
        model or {},
        form_data.get('metadata'),
    )
    if is_awg_request:
        if awg_state is None:
            return form_data
        try:
            await _check_awg_memory_permission(user)
        except HTTPException:
            return form_data
        return await _add_awg_preference_context(form_data, user)

    if not model_allows_memory(model):
        return form_data

    user_messages = []
    for message in reversed(form_data.get('messages', [])):
        if message.get('role') != 'user':
            continue

        content = get_content_from_message(message)
        if isinstance(content, str) and content.strip():
            user_messages.append(content.strip())

        if len(user_messages) >= 7:
            break

    query = '\n\n'.join(reversed(user_messages))[-4000:]
    if not query:
        return form_data

    all_memories = await Memories.get_memories_by_user_id(user.id)
    results = None
    try:
        from open_webui.routers.memories import QueryMemoryForm, query_memory

        results = await query_memory(request, QueryMemoryForm(content=query, k=8), user)
    except Exception as e:
        log.debug(e)

    sections = {'user': [], 'neighborhood': [], 'context': []}
    seen_ids = set()
    for memory in sorted(
        [memory for memory in (all_memories or []) if memory.type == 'user'],
        key=lambda item: (item.path or '', item.updated_at or 0, item.id or ''),
    ):
        seen_ids.add(memory.id)
        sections['user'].append(memory_label(memory))

    for hint in memory_path_hints(query, all_memories):
        for memory in search_memory_rows(
            all_memories,
            path=hint,
            memory_type='context',
            limit=4,
        ):
            if memory.id in seen_ids:
                continue
            seen_ids.add(memory.id)
            sections['neighborhood'].append(memory_label(memory))

    if results and hasattr(results, 'documents') and results.documents:
        for doc_idx, doc in enumerate(results.documents[0]):
            if not doc:
                continue

            metadata = {}
            if results.metadatas and results.metadatas[0] and len(results.metadatas[0]) > doc_idx:
                metadata = results.metadatas[0][doc_idx] or {}

            memory_id = None
            if results.ids and results.ids[0] and len(results.ids[0]) > doc_idx:
                memory_id = results.ids[0][doc_idx]
            if memory_id and memory_id in seen_ids:
                continue
            if memory_id:
                seen_ids.add(memory_id)

            content = str(doc)
            if metadata.get('path') and content.startswith(f'{metadata.get("path")}\n'):
                content = content[len(metadata.get('path')) + 1 :]
            label = f'{metadata.get("path")}: {content}' if metadata.get('path') else content
            sections[Memories.normalize_memory_type(metadata.get('type'))].append(label)

    parts = []
    for title, key in (
        ('User Memory', 'user'),
        ('Memory Neighborhood', 'neighborhood'),
        ('Relevant Context', 'context'),
    ):
        if sections[key]:
            ordered = sorted(sections[key], key=lambda memory: (memory.casefold(), memory))
            parts.append(f'[{title}]\n' + '\n'.join(f'- {memory}' for memory in ordered))
    if not parts:
        return form_data

    config = await Config.get_many('memories.user_char_limit', 'memories.context_char_limit')
    try:
        user_limit = max(250, int(config.get('memories.user_char_limit') or 2000))
    except Exception:
        user_limit = 2000
    try:
        context_limit = max(250, int(config.get('memories.context_char_limit') or 2000))
    except Exception:
        context_limit = 2000

    messages = form_data['messages']
    if messages and messages[0].get('role') == 'system':
        content = messages[0].get('content', '')
        if isinstance(content, str) and MEMORY_CONTEXT_OPEN in content:
            start = content.find(MEMORY_CONTEXT_OPEN)
            end = content.find(MEMORY_CONTEXT_CLOSE, start)
            if end != -1:
                messages[0]['content'] = (content[:start] + content[end + len(MEMORY_CONTEXT_CLOSE) :]).strip()

    user_parts = [part for part in parts if part.startswith('[User Memory]')]
    context_parts = [part for part in parts if not part.startswith('[User Memory]')]
    rendered = '\n\n'.join(
        [
            '\n\n'.join(user_parts)[:user_limit],
            '\n\n'.join(context_parts)[:context_limit],
        ]
    ).strip()
    if not rendered:
        return form_data

    memory_context = f'{MEMORY_CONTEXT_OPEN}\n{rendered}\n{MEMORY_CONTEXT_CLOSE}'
    form_data['messages'] = add_or_update_system_message(memory_context, messages, append=True)
    return form_data


async def review_memory_after_turn(
    *,
    request,
    user,
    model: dict | None,
    metadata: dict,
    form_data: dict,
    assistant_message: dict,
    messages: list[dict],
) -> None:
    if not model_allows_memory(model):
        return
    is_awg_request, _ = get_awg_request_state(request, model or {}, metadata)
    if is_awg_request:
        return

    features = metadata.get('features') or {}
    if not features.get('memory'):
        return

    assistant_content = get_content_from_message(assistant_message)
    if not isinstance(assistant_content, str) or not assistant_content.strip():
        return

    config = await Config.get_many(
        'memories.background_review.enable',
        'memories.review_interval_turns',
    )
    if not config.get('memories.background_review.enable'):
        return

    try:
        interval = max(1, int(config.get('memories.review_interval_turns', 10)))
    except Exception:
        interval = 10

    user_turns = len([message for message in messages if message.get('role') == 'user'])
    if user_turns == 0 or user_turns % interval != 0:
        return

    task = asyncio.create_task(
        _review_memory(
            request=request,
            user=user,
            model=model,
            metadata=metadata,
            form_data=form_data,
            assistant_message=assistant_message,
            messages=messages,
        )
    )

    def log_failure(done_task):
        try:
            done_task.result()
        except Exception as e:
            log.debug('Memory review failed: %s', e)

    task.add_done_callback(log_failure)


async def _review_memory(
    *,
    request,
    user,
    model: dict | None,
    metadata: dict,
    form_data: dict,
    assistant_message: dict,
    messages: list[dict],
) -> None:
    existing_memories = await Memories.get_memories_by_user_id(user.id)
    existing_lines = [
        f'- id={memory.id} type={memory.type} path={memory.path or ""} content={memory.content}'
        for memory in (existing_memories or [])[:80]
    ]

    assistant_content = get_content_from_message(assistant_message)
    if not isinstance(assistant_content, str):
        assistant_content = ''

    transcript_lines = []
    for message in messages[-16:]:
        role = message.get('role', '')
        content = message.get('content', '')
        if not isinstance(content, str):
            content = get_content_from_message(message)
        content = content.strip()
        if role not in {'user', 'assistant'} or not content:
            continue
        if len(content) > 1600:
            content = f'{content[:1000]}\n...(truncated)...\n{content[-400:]}'
        transcript_lines.append(f'{role}: {content}')

    if assistant_content.strip():
        assistant_final = assistant_content.strip()
        if len(assistant_final) > 1600:
            assistant_final = f'{assistant_final[:1000]}\n...(truncated)...\n{assistant_final[-400:]}'
        transcript_lines.append(f'assistant_final: {assistant_final}')

    model_id = model.get('id') if isinstance(model, dict) else form_data.get('model')
    operations = await _generate_memory_operations(
        request=request,
        user=user,
        model_id=model_id,
        metadata=metadata,
        existing_text='\n'.join(existing_lines) if existing_lines else '(none)',
        transcript='\n\n'.join(transcript_lines),
    )
    if operations:
        from open_webui.routers.memories import UpdateMemoriesForm, update_memories

        await update_memories(request, UpdateMemoriesForm(operations=operations, source='background_review'), user)


async def _generate_memory_operations(
    *,
    request,
    user,
    model_id: str,
    metadata: dict,
    existing_text: str,
    transcript: str,
) -> list[dict[str, Any]]:
    from open_webui.utils.chat import generate_chat_completion

    review_prompt = f"""Review the completed conversation turn and decide whether long-term memory should change.

Memory types:
- user: durable facts, preferences, or instructions about the user.
- context: other durable context that may help future chats for this user account.

Rules:
- Save enduring details that can improve future conversations.
- Do not save one-off activity, meals, temporary mood, routine daily events, or other short-lived details unless the user explicitly asks to remember them.
- Do not save secrets, credentials, transient task steps, or unsupported guesses.
- Use path when there is a clear path for the memory.
- Leave path empty when there is no clear place for the memory.
- Prefer replace/move/remove over duplicate add when an existing memory should change.
- Do not invent type, status, trait, score, importance, or stability schemas.
- Return only JSON in this shape:
  {{"operations":[
    {{"action":"add","type":"user|context","path":"...","content":"..."}},
    {{"action":"replace","id":"...","type":"user|context","path":"...","content":"..."}},
    {{"action":"move","id":"...","path":"..."}},
    {{"action":"remove","id":"..."}}
  ]}}
- Use an empty operations array if nothing should be remembered.

Existing memories:
{existing_text}

Conversation:
{transcript}
"""

    response = await generate_chat_completion(
        request,
        form_data={
            'model': model_id,
            'messages': [
                {
                    'role': 'system',
                    # LICENSE covers this Open WebUI system identifier.
                    # Do not alter, remove, obscure, or replace it except as LICENSE permits:
                    # https://docs.openwebui.com/license.
                    'content': "You are Open WebUI's private memory reviewer. Return only valid JSON.",
                },
                {'role': 'user', 'content': review_prompt},
            ],
            'stream': False,
            'metadata': {
                'task': 'memory_review',
                'chat_id': metadata.get('chat_id'),
                'message_id': metadata.get('message_id'),
            },
        },
        user=user,
    )

    if not isinstance(response, dict) or not response.get('choices'):
        return []

    response_message = response.get('choices', [{}])[0].get('message', {})
    content = response_message.get('content') or response_message.get('reasoning_content') or ''
    start = content.find('{')
    end = content.rfind('}')
    if start == -1 or end == -1 or end < start:
        return []

    try:
        parsed = JSONCodec.loads(content[start : end + 1])
    except Exception:
        return []

    operations = parsed.get('operations') if isinstance(parsed, dict) else None
    return operations if isinstance(operations, list) else []
