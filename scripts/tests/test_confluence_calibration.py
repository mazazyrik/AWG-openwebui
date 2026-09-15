import ast
import asyncio
import importlib.util
import json
import logging
import re
import sqlite3
import stat
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from open_webui.integrations.confluence.grounding_filter import CITATION_FAILURE, CLARIFY, UNAVAILABLE, UNKNOWN

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('confluence_calibration', ROOT / 'scripts/confluence_calibration.py')
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)


def test_partial_corpus_distinguishes_roster_from_count_and_registry_questions():
    cases = json.loads((ROOT / 'scripts/confluence_calibration_cases.json').read_text())
    partial = {case['id']: case for case in cases if case['family'] == 'partial'}
    assert len(partial) == 6
    for case_id in ('partial-6',):
        expected = partial[case_id]['expected']
        assert expected['kind'] == 'safe_unknown'
        assert expected['facts'] == []
        assert expected['source_ids'] == ['S1']
        assert calibration.evaluate(UNKNOWN, expected, partial[case_id]['lookup']['results'], 'unknown')['passed']
    for case_id in ('partial-1', 'partial-2', 'partial-3', 'partial-5'):
        expected = partial[case_id]['expected']
        assert expected['kind'] == 'partial'
        assert expected['facts']
        assert expected['source_ids']
        assert not calibration.evaluate(UNKNOWN, expected, partial[case_id]['lookup']['results'], 'unknown')['passed']


    count_case = partial['partial-4']
    expected = count_case['expected']
    assert expected['kind'] == 'safe_unknown'
    assert expected['facts'] == []
    assert expected['source_ids'] == ['S1']
    sources = count_case['lookup']['results']
    qualified = f'По этому источнику общее количество определить нельзя [S1] {sources[0]["url"]}'
    assert calibration.evaluate(qualified, expected, sources, 'answer')['passed']
    assert not calibration.evaluate(qualified.replace('[S1]', ''), expected, sources, 'answer')['passed']
    assert not calibration.evaluate(qualified.replace(sources[0]['url'], ''), expected, sources, 'answer')['passed']
    assert calibration.evaluate(UNKNOWN, expected, sources, 'unknown')['passed']
    assert not calibration.evaluate(f'[S1] {sources[0]["url"]}', expected, sources, 'answer')['passed']


def extracted_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    definitions = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize(
    ('answer', 'kind', 'expected'),
    [
        (CITATION_FAILURE, 'citation_failure', 'unknown'),
        (CITATION_FAILURE, 'citation_failure', 'clarification'),
        (UNKNOWN, 'unknown', 'clarification'),
        (UNAVAILABLE, 'unavailable', 'unknown'),
    ],
)
def test_technical_fallback_and_safe_outcomes_cannot_be_confused(answer, kind, expected):
    result = calibration.evaluate(answer, {'kind': expected, 'facts': [], 'forbidden': [], 'source_ids': []}, [], kind)
    assert result['passed'] is False
    assert result['expected_outcome'] is False


def test_correct_clarification_passes_only_clarification():
    result = calibration.evaluate(
        CLARIFY, {'kind': 'clarification', 'facts': [], 'forbidden': [], 'source_ids': []}, [], 'clarification'
    )
    assert result['passed'] is True


def test_original_unverified_response_is_not_an_absence_result():
    text = 'Не могу дать подтвержденный ответ: найденные сведения не были корректно процитированы.'
    instance = SimpleNamespace()
    assert calibration.baseline_response_kind(text, instance) == 'citation_failure'


def test_native_citations_preserve_external_url_without_local_file_route():
    namespace = extracted_functions(
        ROOT / 'backend/open_webui/utils/middleware.py',
        {'get_citation_source_from_tool_result'},
        {'JSONCodec': json, 'log': logging.getLogger(__name__)},
    )
    chunks = [
        {
            'file_id': 'confluence-123',
            'type': 'external',
            'source': 'Team',
            'content': 'One',
            'url': 'https://confluence.example.com/p/123',
            'page_id': '123',
            'version': 9,
            'hash': 'hash',
            'space': 'YANDEX',
        },
        {
            'file_id': 'confluence-123',
            'type': 'external',
            'source': 'Team',
            'content': 'Two',
            'url': 'https://confluence.example.com/p/123',
            'page_id': '123',
        },
        {'file_id': 'local', 'source': 'Local', 'content': 'Local content'},
    ]
    result = namespace['get_citation_source_from_tool_result']('query_knowledge_files', {}, json.dumps(chunks))
    assert len(result) == 2
    assert result[0]['source']['url'] == chunks[0]['url']
    assert result[0]['document'] == ['One', 'Two']
    assert result[0]['metadata'][0]['version'] == 9
    assert result[0]['metadata'][0]['hash'] == 'hash'
    assert 'file_id' not in result[0]['metadata'][0]
    assert result[1]['metadata'][0]['file_id'] == 'local'


@pytest.mark.asyncio
@pytest.mark.parametrize('allow_external', [True, False])
async def test_mixed_grep_reports_only_accessible_external_bases(monkeypatch, allow_external):
    file = SimpleNamespace(id='local-file', filename='team.txt', data={'content': 'alpha developer'})
    local = SimpleNamespace(id='local', user_id='user', meta={})
    external = SimpleNamespace(id='external', user_id='other', meta={'source': 'external'})
    files_api = SimpleNamespace(get_file_by_id=AsyncMock(return_value=file))
    knowledge_api = SimpleNamespace(
        get_knowledge_by_id=AsyncMock(side_effect=[local, external]), get_files_by_id=AsyncMock(return_value=[file])
    )
    access_api = SimpleNamespace(has_access=AsyncMock(return_value=allow_external))
    for name, symbol, value in [
        ('files', 'Files', files_api),
        ('knowledge', 'Knowledges', knowledge_api),
        ('access_grants', 'AccessGrants', access_api),
    ]:
        module = ModuleType(f'open_webui.models.{name}')
        setattr(module, symbol, value)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    matcher_module = ModuleType('open_webui.tools.knowledge_fs')
    matcher_namespace = extracted_functions(
        ROOT / 'backend/open_webui/tools/knowledge_fs.py', {'build_matcher', 'is_regex_pattern'}, {'re': re}
    )
    matcher_module.build_matcher = matcher_namespace['build_matcher']
    monkeypatch.setitem(sys.modules, matcher_module.__name__, matcher_module)
    namespace = extracted_functions(
        ROOT / 'backend/open_webui/tools/builtin.py',
        {'grep_knowledge_files', '_grep_file_models'},
        {
            'JSONCodec': json,
            'log': logging.getLogger(__name__),
            'asyncio': asyncio,
            'Optional': Optional,
            'Request': object,
            'KNOWLEDGE_GREP_MAX_MATCHES': 100,
            'Groups': SimpleNamespace(get_groups_by_member_id=AsyncMock(return_value=[])),
        },
    )
    result = await namespace['grep_knowledge_files'](
        'alpha',
        __request__=object(),
        __user__={'id': 'user'},
        __model_knowledge__=[{'type': 'collection', 'id': 'local'}, {'type': 'collection', 'id': 'external'}],
    )
    if allow_external:
        parsed = json.loads(result)
        assert 'alpha developer' in parsed['local_results']
        assert parsed['knowledge_ids'] == ['external']
        assert 'not searched' in parsed['notice']
    else:
        assert 'alpha developer' in result
        assert 'external' not in result
    assert knowledge_api.get_files_by_id.await_count == 1
    access_api.has_access.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('temperature', [None, 0.0])
async def test_incomplete_model_answer_cannot_pass_calibration(tmp_path, monkeypatch, temperature):
    candidate = ROOT / 'backend/open_webui/integrations/confluence/grounding_filter.py'
    baseline = """from pydantic import BaseModel
def collect_sources(payloads):
    return payloads[0]['results']
class Filter:
    class Valves(BaseModel):
        lookup_url: str = ''
        admin_token: str = ''
    def __init__(self):
        self.valves = self.Valves()
"""
    database = tmp_path / 'fixture.db'
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE function (id TEXT, content TEXT, valves TEXT)')
        connection.execute('CREATE TABLE skill (id TEXT, content TEXT)')
        connection.execute('CREATE TABLE model (id TEXT, params TEXT)')
        connection.execute('INSERT INTO function VALUES (?, ?, ?)', ('test', baseline, '{}'))
        connection.execute('INSERT INTO skill VALUES (?, ?)', ('test', ''))
        connection.execute('INSERT INTO model VALUES (?, ?)', ('test', '{}'))
    url = 'https://confluence.example.com/p/123'
    payload = {'found': True, 'mode': 'index', 'results': [{'page_id': '123', 'url': url, 'text': 'Person'}]}
    cases = tmp_path / 'cases.json'
    cases.write_text(
        json.dumps(
            [
                {
                    'id': 'one',
                    'family': 'people',
                    'messages': [{'role': 'user', 'content': 'Who?'}],
                    'lookup': payload,
                    'expected': {'kind': 'answer', 'facts': ['Person'], 'forbidden': [], 'source_ids': ['S1']},
                }
            ]
        )
    )
    prompt = tmp_path / 'prompt.md'
    prompt.write_text('')
    original = httpx.AsyncClient

    def transport(request):
        if request.url.path == '/lookup':
            return httpx.Response(200, json=payload)
        sent_payload = json.loads(request.content)
        if temperature is None:
            assert 'temperature' not in sent_payload
        else:
            assert sent_payload['temperature'] == temperature
        sent = sent_payload['messages']
        assert sent[0]['role'] == 'system'
        assert sum(item['role'] == 'system' for item in sent) == 1
        return httpx.Response(
            200,
            json={
                'choices': [
                    {'message': {'role': 'assistant', 'content': f'Person [S1] {url}'}, 'finish_reason': 'length'}
                ]
            },
        )

    monkeypatch.setattr(
        calibration.httpx, 'AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(transport))
    )
    monkeypatch.setattr(calibration, 'ThreadingHTTPServer', Mock(return_value=Mock(server_port=12345)))
    output = tmp_path / 'output.json'
    await calibration.run(
        SimpleNamespace(
            cases=cases,
            database=database,
            filter_id='test',
            model_id='test',
            candidate=candidate,
            prompt=prompt,
            skill=prompt,
            max_tokens=32,
            disable_thinking=True,
            temperature=temperature,
            model_url='http://model.test/v1',
            output=output,
            transcript=tmp_path / 'private-transcript.jsonl',
        )
    )
    report = json.loads(output.read_text())
    result = next(record for record in report['records'] if record['variant'] == 'candidate')
    assert result['facts_present'] is True
    assert result['citations_present'] is True
    assert result['incomplete'] is True
    assert result['passed'] is False
    assert stat.S_IMODE((tmp_path / 'private-transcript.jsonl').stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ('text', 'passed'),
    [
        ('Это только часть ответа.', False),
        ('В AWG всего 100 разработчиков. Это только часть ответа.', False),
        ('Общее количество определить нельзя, но итого 100.', False),
        ('Общее количество определить нельзя, 100 разработчиков.', False),
        ('Общее количество разработчиков AWG — 100, но точное число определить нельзя.', False),
        ('По этому источнику общее количество определить нельзя.', True),
        ('Источник не подтверждает точное количество разработчиков.', True),
        ('Источник не подтверждает точное число разработчиков.', True),
    ],
)
def test_qualified_unknown_requires_unknown_total_and_rejects_numeric_claims(text, passed):
    source = {'url': 'https://confluence.example.com/p/900001'}
    expected = {'kind': 'safe_unknown', 'subject': 'count', 'facts': [], 'forbidden': [], 'source_ids': ['S1']}
    answer = f'{text} [S1] {source["url"]}'
    assert calibration.evaluate(answer, expected, [source], 'answer')['passed'] is passed


@pytest.mark.parametrize('claim', ['реестр существует', 'реестра не существует', 'реестр отсутствует'])
def test_safe_unknown_rejects_registry_claims_even_with_uncertainty(claim):
    source = {'url': 'https://confluence.example.com/p/900001'}
    expected = {
        'kind': 'safe_unknown', 'subject': 'registry', 'facts': [], 'forbidden': [], 'source_ids': ['S1'],
        'unsafe_claims': ['реестр существует', 'реестра не существует', 'реестр отсутствует'],
    }
    answer = f'Источник не подтверждает наличие реестра. Но {claim}. [S1] {source["url"]}'
    assert not calibration.evaluate(answer, expected, [source], 'answer')['passed']


def test_safe_unknown_registry_answer_requires_real_citation():
    source = {'url': 'https://confluence.example.com/p/900001'}
    expected = {'kind': 'safe_unknown', 'subject': 'registry', 'facts': [], 'forbidden': [], 'source_ids': ['S1']}
    answer = f'Источник не подтверждает наличие реестра [S1] {source["url"]}'
    assert calibration.evaluate(answer, expected, [source], 'answer')['passed']
    assert not calibration.evaluate(answer.replace('[S1]', ''), expected, [source], 'answer')['passed']
    assert not calibration.evaluate(answer.replace(source['url'], ''), expected, [source], 'answer')['passed']


@pytest.mark.parametrize(
    ('subject', 'text', 'kind'),
    [
        ('registry', 'Не нашёл подробностей. Да, такой реестр есть.', 'unknown'),
        ('registry', 'По этому источнику общее количество определить нельзя.', 'answer'),
        ('count', 'Источник не подтверждает наличие реестра.', 'answer'),
    ],
)
def test_safe_unknown_rejects_uncontrolled_unknown_and_wrong_subject(subject, text, kind):
    source = {'url': 'https://confluence.example.com/p/900001'}
    expected = {
        'kind': 'safe_unknown', 'subject': subject, 'facts': [], 'forbidden': [], 'source_ids': ['S1'],
    }
    answer = text if kind == 'unknown' else f'{text} [S1] {source["url"]}'
    assert not calibration.evaluate(answer, expected, [source], kind)['passed']


@pytest.mark.parametrize('value', ['-0.1', '2.1', 'nan', 'inf'])
def test_temperature_argument_rejects_out_of_range(value):
    with pytest.raises(calibration.argparse.ArgumentTypeError):
        calibration.temperature_argument(value)


@pytest.mark.parametrize('value', ['0', '2'])
def test_temperature_argument_accepts_bounds(value):
    assert calibration.temperature_argument(value) == float(value)


def test_historical_hardcoded_host_filter_fails_before_comparison():
    source = """
from urllib.parse import urlsplit

class Filter:
    pass

def collect_sources(payloads):
    sources = []
    for item in payloads[0]['results']:
        parsed = urlsplit(item['url'])
        if parsed.scheme != 'https' or parsed.netloc != 'historical.example.com':
            continue
        sources.append(item)
    return sources
"""
    with pytest.raises(ValueError, match='rejects the synthetic source host; comparison aborted'):
        calibration.load_filter(source, 'historical_host_filter')


def test_current_filter_accepts_synthetic_host_in_calibration():
    source = (ROOT / 'backend/open_webui/integrations/confluence/grounding_filter.py').read_text()
    instance = calibration.load_filter(source, 'current_host_filter')
    module = sys.modules[instance.__class__.__module__]
    assert module.ALLOWED_SOURCE_HOST == 'confluence.example.com'
