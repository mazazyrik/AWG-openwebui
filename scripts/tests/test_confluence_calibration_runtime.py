import importlib.util
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('calibration_runtime', ROOT / 'scripts/confluence_calibration_runtime.py')
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def test_private_snapshot_is_owner_only_and_cannot_overwrite(tmp_path):
    path = tmp_path / 'snapshot.json'
    runtime.write_private_json(path, {'synthetic': True})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text()) == {'synthetic': True}
    with pytest.raises(FileExistsError):
        runtime.write_private_json(path, {'synthetic': False})


def test_saved_judgment_requires_fact_next_to_source_and_aligned_output():
    url = 'https://conf.awg.ru/p/123'
    expected = {'facts': ['Person'], 'source_urls': [url], 'forbidden': ['https://conf.awg.ru/p/456']}
    answer = f'Person [S1] {url}'
    assert runtime.judge_saved_answer({'content': answer}, expected)['passed'] is True
    assert runtime.judge_saved_answer({'content': f'Person\n\n{url}'}, expected)['passed'] is False
    assert runtime.judge_saved_answer({'content': answer + ' https://conf.awg.ru/p/456'}, expected)['passed'] is False
    assert (
        runtime.judge_saved_answer(
            {
                'content': answer,
                'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'old unverified answer'}]}],
            },
            expected,
        )['passed']
        is False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('model_id', ['baseline', 'candidate'])
async def test_saved_turn_reads_persisted_final_instead_of_http_answer(model_id):
    assistant_id = None

    def transport(request):
        nonlocal assistant_id
        if request.url.path == '/api/chat/completions':
            body = json.loads(request.content)
            assistant_id = body['id']
            assert body['parent_id'] is None
            assert body['user_message']['parentId'] is None
            assert body['params']['max_tokens'] == 1024
            assert body['params']['temperature'] == 0.0
            return httpx.Response(200, json={'chat_id': 'owned', 'task_ids': ['task'], 'content': 'not final'})
        if request.url.path == '/api/tasks/chat/owned':
            return httpx.Response(200, json={'task_ids': []})
        if request.url.path == '/api/v1/chats/owned':
            return httpx.Response(
                200, json={'chat': {'history': {'messages': {assistant_id: {'content': 'saved final'}}}}}
            )
        raise AssertionError(request.url.path)

    owned = set()
    async with httpx.AsyncClient(base_url='http://app.test', transport=httpx.MockTransport(transport)) as client:
        chat_id, _, message = await runtime.saved_turn(
            client, model_id, 'question', [], None, None, 'session', 5, owned, temperature=0.0
        )
    assert message['content'] == 'saved final'
    assert chat_id == 'owned'
    assert owned == {'owned'}


@pytest.mark.asyncio
async def test_timeout_stops_only_tasks_created_by_this_turn():
    stopped = []

    def transport(request):
        if request.url.path == '/api/chat/completions':
            return httpx.Response(200, json={'chat_id': 'owned', 'task_ids': ['own-task']})
        if request.url.path.startswith('/api/tasks/stop/'):
            stopped.append(request.url.path.rsplit('/', 1)[1])
            return httpx.Response(200, json={})
        raise AssertionError(request.url.path)

    async with httpx.AsyncClient(base_url='http://app.test', transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(TimeoutError):
            await runtime.saved_turn(client, 'model', 'question', [], None, None, 'session', 0, set())
    assert stopped == ['own-task']


def test_nonempty_structured_output_without_final_text_is_not_aligned():
    expected = {'facts': ['Person'], 'source_urls': ['https://conf.awg.ru/p/123']}
    result = runtime.judge_saved_answer(
        {'content': 'Person https://conf.awg.ru/p/123', 'output': [{'type': 'reasoning', 'content': []}]}, expected
    )
    assert result['content_output_aligned'] is False
    assert result['passed'] is False


@pytest.mark.parametrize('kind', ['unknown', 'clarification', 'answer', 'partial'])
def test_saved_technical_refusal_cannot_pass_factless_gold(kind):
    message = {'content': 'Не удалось подтвердить ответ по найденным материалам. Можно уточнить вопрос.'}
    assert runtime.judge_saved_answer(message, {'kind': kind})['passed'] is False


def test_saved_outage_only_passes_expected_outage():
    message = {'content': 'Сейчас не удалось проверить Confluence. Попробуйте ещё раз чуть позже.'}
    assert runtime.judge_saved_answer(message, {'kind': 'unknown'})['passed'] is False
    assert runtime.judge_saved_answer(message, {'kind': 'unavailable'})['passed'] is True


@pytest.mark.asyncio
async def test_clone_resolves_preset_to_physical_model():
    async with httpx.AsyncClient(
        base_url='http://app.test',
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={'data': [{'id': 'physical'}, {'id': 'preset', 'preset': True}]})
        ),
    ) as client:
        assert await runtime.physical_model_id(client, {'base_model_id': 'physical'}, 'preset') == 'physical'
        with pytest.raises(RuntimeError, match='physical model'):
            await runtime.physical_model_id(client, {}, 'preset')


@pytest.mark.asyncio
async def test_followup_persists_user_parent_id():
    assistant_id = None

    def transport(request):
        nonlocal assistant_id
        if request.url.path == '/api/chat/completions':
            body = json.loads(request.content)
            assistant_id = body['id']
            assert body['parent_id'] == 'previous-assistant'
            assert body['user_message']['parentId'] == 'previous-assistant'
            return httpx.Response(200, json={'chat_id': 'owned', 'task_ids': ['task']})
        if request.url.path == '/api/tasks/chat/owned':
            return httpx.Response(200, json={'task_ids': []})
        return httpx.Response(200, json={'chat': {'history': {'messages': {assistant_id: {'content': 'final'}}}}})

    async with httpx.AsyncClient(base_url='http://app.test', transport=httpx.MockTransport(transport)) as client:
        await runtime.saved_turn(
            client, 'model', 'question', [], 'owned', 'previous-assistant', 'session', 5, {'owned'}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize('transport_error', [False, True])
async def test_poll_failure_stops_owned_tasks(transport_error):
    stopped = []

    def transport(request):
        if request.url.path == '/api/chat/completions':
            return httpx.Response(200, json={'chat_id': 'owned', 'task_ids': ['own-task']})
        if request.url.path == '/api/tasks/chat/owned':
            if transport_error:
                raise httpx.ConnectError('unavailable', request=request)
            return httpx.Response(500, json={})
        if request.url.path.startswith('/api/tasks/stop/'):
            stopped.append(request.url.path.rsplit('/', 1)[1])
            return httpx.Response(200, json={})
        raise AssertionError(request.url.path)

    async with httpx.AsyncClient(base_url='http://app.test', transport=httpx.MockTransport(transport)) as client:
        with pytest.raises((RuntimeError, httpx.ConnectError)):
            await runtime.saved_turn(client, 'model', 'question', [], None, None, 'session', 5, set())
    assert stopped == ['own-task']


@pytest.mark.asyncio
async def test_archive_failure_still_writes_metrics_and_attempts_every_owned_chat(tmp_path):
    attempted = []

    def transport(request):
        attempted.append(request.url.path)
        return httpx.Response(500, json={})

    async with httpx.AsyncClient(base_url='http://app.test', transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(RuntimeError, match='metrics were saved'):
            await runtime.finalize_run(client, {'one', 'two'}, tmp_path, 'run', 'model', [{'passed': False}])
    assert len(attempted) == 2
    metrics = json.loads((tmp_path / 'metrics.json').read_text())
    assert set(metrics['archive_failures']) == {'one', 'two'}
    assert metrics['results'] == [{'passed': False}]


def test_production_candidate_preserves_all_original_decoding_params():
    original = {
        'system': 'old',
        'temperature': 0.3,
        'max_tokens': 4096,
        'custom_params': {'chat_template_kwargs': {'enable_thinking': True}, 'other': 42},
    }
    result = runtime.candidate_params(original, 'new', True)
    assert result == {**original, 'system': 'new'}
    result['custom_params']['other'] = 10
    assert original['custom_params']['other'] == 42


@pytest.mark.asyncio
async def test_production_saved_turn_omits_per_request_decoding_overrides():
    assistant_id = None

    def transport(request):
        nonlocal assistant_id
        if request.url.path == '/api/chat/completions':
            body = json.loads(request.content)
            assistant_id = body['id']
            assert 'params' not in body
            assert 'max_tokens' not in body
            assert 'chat_template_kwargs' not in body
            return httpx.Response(200, json={'chat_id': 'owned', 'task_ids': ['task']})
        if request.url.path == '/api/tasks/chat/owned':
            return httpx.Response(200, json={'task_ids': []})
        return httpx.Response(200, json={'chat': {'history': {'messages': {assistant_id: {'content': 'final'}}}}})

    async with httpx.AsyncClient(base_url='http://app.test', transport=httpx.MockTransport(transport)) as client:
        await runtime.saved_turn(
            client, 'model', 'question', [], None, None, 'session', 5, set(), production_settings=True
        )


def test_controlled_candidate_temperature_is_explicit_and_production_preserves_original():
    assert runtime.candidate_params({'temperature': 0.8}, 'new', False, 0)['temperature'] == 0
    assert runtime.candidate_params({'temperature': 0.8}, 'new', True)['temperature'] == 0.8


@pytest.mark.parametrize('value', ['-1', '3', 'nan', 'inf'])
def test_runtime_temperature_argument_rejects_nonfinite_or_out_of_range(value):
    with pytest.raises(runtime.argparse.ArgumentTypeError):
        runtime.temperature_argument(value)


def test_runtime_cli_rejects_temperature_in_production_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'runtime',
            '--model-id',
            'model',
            '--candidate',
            'filter',
            '--prompt',
            'prompt',
            '--skill',
            'skill',
            '--gold',
            'gold',
            '--production-settings',
            '--temperature',
            '0',
        ],
    )
    with pytest.raises(SystemExit) as result:
        runtime.main()
    assert result.value.code == 2


@pytest.mark.asyncio
async def test_controlled_run_updates_candidate_valve_to_request_temperature(tmp_path, monkeypatch):
    recorded = []
    source = tmp_path / 'source'
    source.write_text('synthetic')
    original_valves = {'temperature': 0.8, 'always_lookup': True}

    async def fake_api(client, method, path, **kwargs):
        if path.endswith('/valves/update'):
            recorded.append(kwargs['json'])
            raise RuntimeError('stop after valve verification')
        if path.endswith('/valves'):
            return original_valves
        if path == '/api/v1/functions/create':
            return {'is_global': False, 'is_active': True}
        return {}

    async def physical(*args):
        return 'physical'

    monkeypatch.setattr(runtime, 'BACKUP_ROOT', tmp_path)
    monkeypatch.setattr(runtime, 'server_token', lambda database: 'synthetic')
    monkeypatch.setattr(runtime, 'api', fake_api)
    monkeypatch.setattr(runtime, 'physical_model_id', physical)
    with pytest.raises(RuntimeError, match='stop after valve verification'):
        await runtime.run(
            SimpleNamespace(
                api_url='http://app.test',
                database=tmp_path / 'db',
                filter_id='original',
                model_id='model',
                candidate=source,
                temperature=0.5,
                production_settings=False,
            )
        )
    assert recorded == [{'temperature': 0.5, 'always_lookup': True}]
    assert original_valves['temperature'] == 0.8


@pytest.mark.asyncio
async def test_repeated_runs_create_distinct_skill_names_from_run_identity(tmp_path, monkeypatch):
    names = []
    source = tmp_path / 'source'
    source.write_text('synthetic')

    async def fake_api(client, method, path, **kwargs):
        if path == '/api/v1/skills/create':
            payload = kwargs['json']
            run_id = payload['id'].removeprefix('awg-calibration-')
            assert payload['name'] == f'Confluence calibration {run_id}'
            names.append(payload['name'])
            raise RuntimeError('stop after skill verification')
        if path == '/api/v1/functions/create':
            return {'is_global': False, 'is_active': True}
        return {}

    async def physical(*args):
        return 'physical'

    monkeypatch.setattr(runtime, 'BACKUP_ROOT', tmp_path)
    monkeypatch.setattr(runtime, 'server_token', lambda database: 'synthetic')
    monkeypatch.setattr(runtime, 'api', fake_api)
    monkeypatch.setattr(runtime, 'physical_model_id', physical)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='stop after skill verification'):
            await runtime.run(
                SimpleNamespace(
                    api_url='http://app.test',
                    database=tmp_path / 'db',
                    filter_id='original',
                    model_id='model',
                    candidate=source,
                    skill=source,
                    temperature=0.0,
                    production_settings=False,
                )
            )
    assert len(set(names)) == 2


@pytest.mark.parametrize('marker', ['', '[S2]'])
def test_optional_marker_requirement_applies_next_to_fact_and_source(marker):
    url = 'https://conf.awg.ru/p/123'
    expected = {'kind': 'answer', 'facts': ['Person'], 'source_urls': [url], 'citation_marker_required': True}
    result = runtime.judge_saved_answer({'content': f'Person {marker} {url}'}, expected)
    assert result['passed'] is bool(marker)
