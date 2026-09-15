"""Run synthetic source fixtures through baseline and candidate with the same model."""

import argparse
import asyncio
import copy
import inspect
import json
import os
import re
import sqlite3
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx


class FixtureHandler(BaseHTTPRequestHandler):
    payload = {}
    calls = 0

    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length', 0)))
        type(self).calls += 1
        encoded = json.dumps(type(self).payload).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        pass


def load_filter(source: str, name: str):
    module = ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, name, 'exec'), module.__dict__)
    return module.Filter()


async def invoke(instance, name, body, context):
    handler = getattr(instance, name, None)
    if handler is None:
        return body
    params = inspect.signature(handler).parameters
    result = handler(body=body, **{key: value for key, value in context.items() if key in params})
    return await result if inspect.isawaitable(result) else result


async def fixture_hydration(sources, query):
    return sources, False


def baseline_response_kind(answer: str, instance) -> str:
    module = sys.modules[instance.__class__.__module__]
    rejection = getattr(instance, 'UNVERIFIED_RESPONSE', getattr(module, 'UNVERIFIED_RESPONSE', None))
    lower = answer.strip().casefold()
    if answer.strip() == rejection or 'не были корректно процитированы' in lower:
        return 'citation_failure'
    if any(word in lower for word in ('не удалось проверить', 'недоступ', 'позже')):
        return 'unavailable'
    if any(word in lower for word in ('нет подтверж', 'не удалось подтверд', 'не наш', 'не найден')):
        return 'unknown'
    if 'уточн' in lower or 'какой проект' in lower:
        return 'clarification'
    return 'answer'


def evaluate(answer: str, expected: dict, sources: list[dict], response_kind: str) -> dict:
    lower = answer.casefold()
    facts = all(fact.casefold() in lower for fact in expected['facts'])
    forbidden = any(fact.casefold() in lower for fact in expected['forbidden'])
    citations = all(f'[{sid}]' in answer for sid in expected['source_ids'])
    urls = all(sources[int(sid[1:]) - 1]['url'] in answer for sid in expected['source_ids'])
    citation_required = True
    if expected['kind'] in ('clarification', 'unavailable', 'unknown'):
        outcome = response_kind == expected['kind']
    elif expected['kind'] == 'safe_unknown':
        prose = lower
        for source in sources:
            prose = prose.replace(source['url'].casefold(), '')
        prose = re.sub(r'\[s\d+\]', '', prose)
        unknown_total = bool(
            re.search(
                r'(?:общее|точное|полное)\s+(?:количество|число)[^.\n]{0,80}'
                r'(?:нельзя|невозможно|не уда[её]тся|не позволяет|не подтвержден)',
                prose,
            )
            or re.search(
                r'(?:нельзя|невозможно|не могу|не позволяет|не подтверждает)[^.\n]{0,80}'
                r'(?:общее|точное|полное)\s+(?:количество|число)',
                prose,
            )
        )
        nonconfirmation = unknown_total or bool(
            re.search(
                r'(?:не подтверждает|не подтвержден[оа]?|не позволяет подтвердить|нельзя подтвердить)'
                r'[^.\n]{0,80}(?:реестр|наличие|существование)', prose
            )
            or re.search(r'(?:реестр|наличие|существование)[^.\n]{0,80}(?:не подтвержден|не подтверждает)', prose)
        )
        numeric_total = bool(re.search(r'[0-9]', prose))
        unsafe = any(claim.casefold() in lower for claim in expected.get('unsafe_claims', []))
        controlled_unknown = response_kind == 'unknown' and answer.strip() == (
            'В найденных материалах не удалось подтвердить ответ. '
            'Пришлите ссылку на нужную страницу — проверю её.'
        )
        subject = expected['subject']
        subject_present = (
            any(token in prose for token in ('количеств', 'сколько', 'числ'))
            if subject == 'count' else subject == 'registry' and 'реестр' in prose
        )
        citation_required = not controlled_unknown
        outcome = not numeric_total and not unsafe and (
            controlled_unknown or (response_kind == 'answer' and nonconfirmation and subject_present)
        )
    elif expected['kind'] == 'partial':
        outcome = response_kind == 'answer' and any(
            word in lower for word in ('только', 'часть', 'неполный', 'не полный')
        )
    else:
        outcome = response_kind == 'answer' and bool(expected['facts']) and facts
    return {
        'response_kind': response_kind,
        'facts_present': facts,
        'forbidden_present': forbidden,
        'citations_present': citations and urls,
        'expected_outcome': outcome,
        'passed': facts and not forbidden and (not citation_required or (citations and urls)) and outcome,
    }


async def run(args):
    cases = json.loads(args.cases.read_text())
    connection = sqlite3.connect(f'file:{args.database}?mode=ro', uri=True)
    try:
        row = connection.execute('SELECT content, valves FROM function WHERE id = ?', (args.filter_id,)).fetchone()
        baseline_code = row[0]
        valves = json.loads(row[1] or '{}')
        baseline_skill = connection.execute('SELECT content FROM skill WHERE id = ?', (args.filter_id,)).fetchone()[0]
        parameters = json.loads(
            connection.execute('SELECT params FROM model WHERE id = ?', (args.model_id,)).fetchone()[0]
        )
    finally:
        connection.close()
    candidate_code = args.candidate.read_text()
    configurations = [
        ('baseline', baseline_code, parameters.get('system', ''), baseline_skill),
        ('candidate', candidate_code, args.prompt.read_text(), args.skill.read_text()),
    ]
    decoding = {
        key: parameters[key]
        for key in (
            'temperature',
            'top_p',
            'top_k',
            'min_p',
            'max_tokens',
            'seed',
            'frequency_penalty',
            'presence_penalty',
        )
        if key in parameters and parameters[key] is not None
    }
    decoding['max_tokens'] = args.max_tokens
    if args.temperature is not None:
        decoding['temperature'] = args.temperature
    if args.disable_thinking:
        decoding['chat_template_kwargs'] = {'enable_thinking': False}
    server = ThreadingHTTPServer(('127.0.0.1', 0), FixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    records = []
    headers = {}
    if os.environ.get('CALIBRATION_MODEL_API_KEY'):
        headers['Authorization'] = f'Bearer {os.environ["CALIBRATION_MODEL_API_KEY"]}'
    try:
        async with httpx.AsyncClient(timeout=180, headers=headers) as client:
            for case in cases:
                for variant, code, prompt, skill in configurations:
                    instance = load_filter(code, f'calibration_{variant}')
                    instance.valves = instance.Valves(**valves)
                    instance.valves.lookup_url = f'http://127.0.0.1:{server.server_port}/lookup'
                    instance.valves.admin_token = 'synthetic-fixture'
                    if variant == 'candidate':
                        instance._hydrate_sources = fixture_hydration
                    FixtureHandler.payload = case['lookup']
                    FixtureHandler.calls = 0
                    context = {
                        '__user__': {'id': 'calibration', 'role': 'admin'},
                        '__metadata__': {'chat_id': f'{variant}-{case["id"]}', 'message_id': case['id']},
                        '__request__': SimpleNamespace(state=SimpleNamespace()),
                    }
                    body = {
                        'model': args.model_id,
                        'messages': [{'role': 'system', 'content': f'{prompt}\n\n{skill}'}]
                        + copy.deepcopy(case['messages']),
                    }
                    started = time.monotonic()
                    body = await invoke(instance, 'inlet', body, context)
                    body = await invoke(instance, 'request', body, context)
                    system_content = '\n'.join(
                        item['content'] for item in body['messages'] if item.get('role') == 'system'
                    )
                    provider_messages = [{'role': 'system', 'content': system_content}] + [
                        item for item in body['messages'] if item.get('role') != 'system'
                    ]
                    request_body = {
                        'model': args.model_id,
                        'messages': provider_messages,
                        'stream': False,
                        **decoding,
                    }
                    if 'tool_choice' in body:
                        request_body['tool_choice'] = body['tool_choice']
                    response = await client.post(args.model_url.rstrip('/') + '/chat/completions', json=request_body)
                    response.raise_for_status()
                    choice = response.json()['choices'][0]
                    message = choice['message']
                    original_answer = message.get('content') or ''
                    result = await invoke(
                        instance,
                        'outlet',
                        {'model': args.model_id, 'messages': body['messages'] + [message]},
                        context,
                    )
                    answer = result['messages'][-1].get('content') or ''
                    if variant == 'candidate':
                        state = getattr(context['__request__'].state, 'awg_confluence_grounding', None) or {}
                        response_kind = state.get('outcome', 'state_missing')
                    else:
                        response_kind = baseline_response_kind(answer, instance)
                    record = {
                        'id': case['id'],
                        'family': case['family'],
                        'variant': variant,
                        'latency_seconds': round(time.monotonic() - started, 3),
                        'lookup_calls': FixtureHandler.calls,
                        'tool_calls': len(message.get('tool_calls') or []),
                        'finish_reason': choice.get('finish_reason'),
                        'incomplete': choice.get('finish_reason') != 'stop',
                        **evaluate(answer, case['expected'], case['lookup']['results'], response_kind),
                    }
                    record['passed'] = record['passed'] and not record['incomplete']
                    if args.transcript:
                        descriptor = os.open(args.transcript, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                        with os.fdopen(descriptor, 'a') as transcript:
                            os.fchmod(transcript.fileno(), 0o600)
                            transcript.write(
                                json.dumps(
                                    {
                                        'id': case['id'],
                                        'variant': variant,
                                        'original_answer': original_answer[:32000],
                                        'answer': answer[:32000],
                                    }
                                )
                                + '\n'
                            )
                    records.append(record)
                    print(json.dumps(record), flush=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    report = {
        'mode': 'controlled_synthetic_sources_fully_loaded_skills',
        'judgment': 'lexical_checks_only_semantic_review_required',
        'model_id': args.model_id,
        'decoding': decoding,
        'records': records,
        'summary': {
            variant: {
                'count': sum(record['variant'] == variant for record in records),
                'passed': sum(record['variant'] == variant and record['passed'] for record in records),
            }
            for variant, *_ in configurations
        },
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


def temperature_argument(value: str) -> float:
    temperature = float(value)
    if not 0 <= temperature <= 2:
        raise argparse.ArgumentTypeError('--temperature must be between 0 and 2')
    return temperature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--filter-id', default='awg_confluence_grounding')
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--model-url', required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--prompt', type=Path, required=True)
    parser.add_argument('--skill', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--disable-thinking', action='store_true')
    parser.add_argument('--temperature', type=temperature_argument)
    parser.add_argument('--transcript', type=Path)
    args = parser.parse_args()
    if args.transcript and not args.transcript.resolve().is_relative_to('/app/backend/data/calibration-backups'):
        parser.error('--transcript must remain under /app/backend/data/calibration-backups')
    if not 1 <= args.max_tokens <= 8192:
        parser.error('--max-tokens must be between 1 and 8192')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
