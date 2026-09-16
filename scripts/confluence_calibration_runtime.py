"""Create a private calibration model and inspect saved chat results on the server."""

import argparse
import asyncio
import copy
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

import httpx
import jwt

BACKUP_ROOT = Path('/app/backend/data/calibration-backups')


def write_private_json(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as output:
        json.dump(value, output, ensure_ascii=False)
        output.write('\n')


def server_token(database: Path) -> str:
    with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as connection:
        row = connection.execute(
            'SELECT id FROM user WHERE role = ? ORDER BY created_at LIMIT 1', ('admin',)
        ).fetchone()
    if not row:
        raise RuntimeError('Calibration requires an existing server administrator')
    secret = os.getenv('WEBUI_SECRET_KEY')
    if not secret:
        secret = Path(os.getenv('WEBUI_SECRET_KEY_FILE', '/app/backend/.webui_secret_key')).read_text().strip()
    now = int(time.time())
    return jwt.encode({'id': row[0], 'jti': str(uuid4()), 'iat': now, 'exp': now + 7200}, secret, algorithm='HS256')


async def api(client: httpx.AsyncClient, method: str, path: str, **kwargs):
    response = await client.request(method, path, **kwargs)
    if response.is_error:
        raise RuntimeError(f'Calibration API {method} {path} returned HTTP {response.status_code}')
    return response.json()


def output_text(message: dict) -> str:
    return ''.join(
        content.get('text', '')
        for item in message.get('output', [])
        if item.get('type') == 'message'
        for content in item.get('content', [])
        if content.get('type') == 'output_text'
    )


def judge_saved_answer(message: dict, expected: dict) -> dict:
    content = message.get('content') or ''
    structured = output_text(message)
    lower = content.casefold()
    facts = all(fact.casefold() in lower for fact in expected.get('facts', []))
    prohibited = any(value.casefold() in lower for value in expected.get('forbidden', []))
    urls = all(url in content for url in expected.get('source_urls', []))
    evidence_together = all(
        any(
            fact.casefold() in paragraph.casefold()
            and any(url in paragraph for url in expected.get('source_urls', []))
            and (not expected.get('citation_marker_required') or bool(re.search(r'\[S[1-9]\d*\]', paragraph)))
            for paragraph in content.split('\n\n')
        )
        for fact in expected.get('facts', [])
    )
    coverage = expected.get('kind') != 'partial' or any(
        phrase in lower for phrase in ('только', 'часть', 'не полный', 'неполный', 'уточните')
    )
    aligned = not message.get('output') or bool(structured) and structured == content
    safe_kinds = {
        (
            'В найденных материалах не удалось подтвердить ответ. Пришлите ссылку на нужную страницу — проверю её.'
        ): 'unknown',
        'Уточните, какой проект или команду вы имеете в виду.': 'clarification',
        'Сейчас не удалось проверить Confluence. Попробуйте ещё раз чуть позже.': 'unavailable',
    }
    response_kind = safe_kinds.get(content.strip(), 'answer')
    if 'не удалось подтвердить ответ по найденным материалам' in lower or 'не были корректно процитированы' in lower:
        response_kind = 'technical_failure'
    kind = expected.get('kind', 'answer')
    if kind in ('unknown', 'clarification', 'unavailable'):
        outcome = response_kind == kind
    else:
        outcome = kind in ('answer', 'partial') and response_kind == 'answer' and bool(expected.get('facts'))
    return {
        'response_kind': response_kind,
        'expected_outcome': outcome,
        'facts_present': facts,
        'forbidden_present': prohibited,
        'sources_present': urls,
        'facts_with_sources': evidence_together,
        'coverage_qualified': coverage,
        'content_output_aligned': aligned,
        'passed': outcome and facts and not prohibited and urls and evidence_together and coverage and aligned,
    }


async def cancel_owned_tasks(client, tasks):
    failures = []
    for task_id in tasks:
        try:
            response = await client.post(f'/api/tasks/stop/{task_id}')
            if response.status_code not in (200, 404):
                failures.append(task_id)
        except httpx.HTTPError:
            failures.append(task_id)
    if failures:
        raise RuntimeError(f'Could not stop {len(failures)} owned calibration tasks')


async def finalize_run(client, owned_chats, directory, run_id, model_id, results):
    failures = []
    for chat_id in owned_chats:
        try:
            await api(client, 'POST', f'/api/v1/chats/{chat_id}/archive')
        except (httpx.HTTPError, RuntimeError, ValueError):
            failures.append(chat_id)
    write_private_json(
        directory / 'metrics.json',
        {
            'run_id': run_id,
            'model_id': model_id,
            'results': results,
            'archive_failures': failures,
        },
    )
    if failures:
        raise RuntimeError(f'Could not archive {len(failures)} owned calibration chats; metrics were saved')


async def saved_turn(
    client,
    model_id,
    text,
    history,
    chat_id,
    parent_id,
    session_id,
    timeout,
    owned_chats,
    production_settings=False,
    temperature=None,
):
    assistant_id = str(uuid4())
    user_message = {'id': str(uuid4()), 'role': 'user', 'content': text, 'parentId': parent_id}
    submitted = await api(
        client,
        'POST',
        '/api/chat/completions',
        json={
            'model': model_id,
            'messages': history + [{'role': 'user', 'content': text}],
            'stream': False,
            'parent_id': parent_id,
            'chat_id': chat_id,
            'user_message': user_message,
            'id': assistant_id,
            'session_id': session_id,
            'background_tasks': {},
            **(
                {}
                if production_settings
                else {
                    'params': {
                        'max_tokens': 1024,
                        **({'temperature': temperature} if temperature is not None else {}),
                        'custom_params': {'chat_template_kwargs': {'enable_thinking': False}},
                    }
                }
            ),
        },
    )
    new_chat_id = submitted.get('chat_id')
    tasks = set(submitted.get('task_ids') or [])
    completed = False
    try:
        if not new_chat_id or not tasks:
            raise RuntimeError('Calibration chat did not create a saved background task')
        if not chat_id:
            owned_chats.add(new_chat_id)
        elif new_chat_id != chat_id:
            raise RuntimeError('Calibration follow-up changed its chat identity')
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            active = await api(client, 'GET', f'/api/tasks/chat/{new_chat_id}')
            if not tasks.intersection(active.get('task_ids') or []):
                saved = await api(client, 'GET', f'/api/v1/chats/{new_chat_id}')
                message = ((saved.get('chat') or {}).get('history') or {}).get('messages', {}).get(assistant_id)
                if not message:
                    raise RuntimeError('Completed calibration task has no saved assistant message')
                completed = True
                return new_chat_id, assistant_id, message
            await asyncio.sleep(2)
        raise TimeoutError('Calibration saved-chat task exceeded the time budget')
    finally:
        if not completed:
            await cancel_owned_tasks(client, tasks)


def candidate_params(original, system, production_settings, temperature=None):
    params = copy.deepcopy(original)
    params['system'] = system
    if not production_settings:
        params.update({'max_tokens': 1024, 'custom_params': {'chat_template_kwargs': {'enable_thinking': False}}})
        if temperature is not None:
            params['temperature'] = temperature
    return params


async def physical_model_id(client, original_model, requested_id):
    model_id = original_model.get('base_model_id') or requested_id
    visible = await api(client, 'GET', '/api/models')
    model = next((item for item in visible.get('data', []) if item.get('id') == model_id), None)
    if model is None or model.get('preset'):
        raise RuntimeError('Calibration base must resolve to an available physical model')
    return model_id


async def verify_candidate(client, model_id, function_id, skill_id):
    visible = await api(client, 'GET', '/api/models')
    candidate = next((item for item in visible.get('data', []) if item.get('id') == model_id), None)
    if candidate is None:
        raise RuntimeError('Calibration model is not available after registration')
    effective_meta = (candidate.get('info') or {}).get('meta') or {}
    if (effective_meta.get('filterIds'), effective_meta.get('skillIds')) != ([function_id], [skill_id]):
        raise RuntimeError('Calibration model did not isolate its filter and skill')
    registered = await api(client, 'GET', '/api/v1/models/model', params={'id': model_id})
    if registered.get('access_grants'):
        raise RuntimeError('Calibration model unexpectedly has sharing grants')


async def run(args):
    run_id = uuid4().hex
    directory = BACKUP_ROOT / run_id
    directory.mkdir(parents=True, mode=0o700)
    os.chmod(directory, 0o700)
    model_id = f'awg-calibration-{run_id}'
    function_id = f'awg_calibration_{run_id}'
    skill_id = f'awg-calibration-{run_id}'
    owned_chats = set()
    results = []
    async with httpx.AsyncClient(
        base_url=args.api_url, timeout=60, headers={'Authorization': f'Bearer {server_token(args.database)}'}
    ) as client:
        original_function = await api(client, 'GET', f'/api/v1/functions/id/{args.filter_id}')
        original_valves = await api(client, 'GET', f'/api/v1/functions/id/{args.filter_id}/valves')
        original_skill = await api(client, 'GET', f'/api/v1/skills/id/{args.filter_id}')
        original_model = await api(client, 'GET', '/api/v1/models/model', params={'id': args.model_id})
        base_model_id = await physical_model_id(client, original_model, args.model_id)
        write_private_json(
            directory / 'baseline.json',
            {
                'function': original_function,
                'valves': original_valves,
                'skill': original_skill,
                'model': original_model,
            },
        )
        write_private_json(directory / 'resource-function.json', {'id': function_id, 'state': 'creation_requested'})
        created_function = await api(
            client,
            'POST',
            '/api/v1/functions/create',
            json={
                'id': function_id,
                'name': 'Confluence calibration',
                'content': args.candidate.read_text(),
                'meta': {},
            },
        )
        if created_function.get('is_global'):
            raise RuntimeError('Calibration function unexpectedly became global')
        candidate_valves = copy.deepcopy(original_valves)
        if not args.production_settings and args.temperature is not None:
            candidate_valves['temperature'] = args.temperature
        await api(client, 'POST', f'/api/v1/functions/id/{function_id}/valves/update', json=candidate_valves)
        if not created_function.get('is_active'):
            await api(client, 'POST', f'/api/v1/functions/id/{function_id}/toggle')
        write_private_json(directory / 'resource-skill.json', {'id': skill_id, 'state': 'creation_requested'})
        await api(
            client,
            'POST',
            '/api/v1/skills/create',
            json={
                'id': skill_id,
                'name': f'Confluence calibration {run_id}',
                'description': 'Private manager-answer calibration',
                'content': args.skill.read_text(),
                'meta': {},
                'access_grants': [],
                'is_active': True,
            },
        )
        meta = copy.deepcopy(original_model.get('meta') or {})
        meta.update({'filterIds': [function_id], 'skillIds': [skill_id]})
        params = candidate_params(
            original_model.get('params') or {}, args.prompt.read_text(), args.production_settings, args.temperature
        )
        write_private_json(directory / 'resource-model.json', {'id': model_id, 'state': 'creation_requested'})
        await api(
            client,
            'POST',
            '/api/v1/models/create',
            json={
                'id': model_id,
                'base_model_id': base_model_id,
                'name': 'Confluence calibration',
                'meta': meta,
                'params': params,
                'access_grants': [],
                'is_active': True,
            },
        )
        await verify_candidate(client, model_id, function_id, skill_id)
        write_private_json(
            directory / 'resources.json', {'model_id': model_id, 'function_id': function_id, 'skill_id': skill_id}
        )
        try:
            cases = json.loads(args.gold.read_text())
            variants = (
                [('baseline', args.model_id), ('candidate', model_id)]
                if args.include_baseline
                else [('candidate', model_id)]
            )
            for case in cases:
                for variant, target in variants:
                    history = []
                    chat_id = None
                    parent_id = None
                    for index, text in enumerate(case['questions']):
                        started = time.monotonic()
                        try:
                            chat_id, parent_id, message = await saved_turn(
                                client,
                                target,
                                text,
                                history,
                                chat_id,
                                parent_id,
                                run_id,
                                args.timeout,
                                owned_chats,
                                production_settings=args.production_settings,
                                temperature=args.temperature,
                            )
                        except TimeoutError:
                            result = {
                                'case': case['id'],
                                'variant': variant,
                                'turn': index,
                                'passed': False,
                                'timeout': True,
                            }
                            results.append(result)
                            print(json.dumps(result), flush=True)
                            break
                        history.extend(
                            [
                                {'role': 'user', 'content': text},
                                {'role': 'assistant', 'content': message.get('content') or ''},
                            ]
                        )
                        write_private_json(
                            directory / f'{case["id"]}-{variant}-{index}.json',
                            {'answer': message.get('content') or '', 'output_text': output_text(message)},
                        )
                        expected = case['expected'][index]
                        result = {
                            'case': case['id'],
                            'variant': variant,
                            'turn': index,
                            'latency_seconds': round(time.monotonic() - started, 3),
                            **judge_saved_answer(message, expected),
                        }
                        results.append(result)
                        print(json.dumps(result), flush=True)
        finally:
            await finalize_run(client, owned_chats, directory, run_id, model_id, results)
    print(
        json.dumps(
            {
                'run_id': run_id,
                'model_id': model_id,
                'settings_mode': 'production' if args.production_settings else 'controlled',
                'temperature': args.temperature,
                'cases': len(results),
                'passed': sum(result['passed'] for result in results),
            }
        ),
        flush=True,
    )


def temperature_argument(value):
    temperature = float(value)
    if not 0 <= temperature <= 2:
        raise argparse.ArgumentTypeError('--temperature must be between 0 and 2')
    return temperature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path('/app/backend/data/webui.db'))
    parser.add_argument('--api-url', default='http://127.0.0.1:8080')
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--filter-id', default='awg_confluence_grounding')
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--prompt', type=Path, required=True)
    parser.add_argument('--skill', type=Path, required=True)
    parser.add_argument('--gold', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--include-baseline', action='store_true')
    parser.add_argument('--production-settings', action='store_true')
    parser.add_argument('--temperature', type=temperature_argument)
    args = parser.parse_args()
    if args.production_settings and args.temperature is not None:
        parser.error('--temperature cannot be combined with --production-settings')
    if not args.gold.resolve().is_relative_to(BACKUP_ROOT):
        parser.error('--gold must remain in the server calibration-backups directory')
    if not 1 <= args.timeout <= 300:
        parser.error('--timeout must be between 1 and 300 seconds')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
