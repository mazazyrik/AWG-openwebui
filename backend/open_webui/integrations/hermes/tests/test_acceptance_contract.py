import json
from pathlib import Path

import pytest
from open_webui.integrations.confluence.scope_router import route_request
from open_webui.integrations.hermes.policy import authorize_tool
from open_webui.integrations.hermes.settings import (
    HERMES_ALLOWED_ARTIFACT_EXTENSIONS,
    HERMES_ALLOWED_ATTACHMENT_EXTENSIONS,
)

CASES_PATH = Path(__file__).parents[5] / 'deploy/hermes/acceptance/cases.json'
CASES = json.loads(CASES_PATH.read_text())
EXPECTED_IDS = {
    'G01',
    'G02',
    'G03',
    'G04',
    'G05',
    'G06',
    'G07',
    'G08',
    'G09',
    'G10',
    'D01',
    'D02',
    'D03',
    'D04',
    'D05',
    'D06',
    'D07',
    'D08',
    'S01',
    'S02',
    'S03',
    'S04',
    'S05',
    'S06',
    'M01',
    'M02',
    'M03',
    'P01',
    'P02',
    'P03',
    'L01',
    'L02',
}
LIVE_IDS = {
    'G03',
    'D01',
    'D02',
    'D03',
    'D04',
    'D05',
    'D06',
    'D07',
    'D08',
    'S01',
    'S02',
    'S03',
    'S04',
    'M01',
    'M02',
    'M03',
    'P01',
    'P02',
    'P03',
    'L01',
    'L02',
}


def test_acceptance_corpus_has_unique_complete_contracts():
    assert {case['id'] for case in CASES} == EXPECTED_IDS
    assert len(CASES) == len(EXPECTED_IDS)
    assert all(set(case) == {'id', 'prompt', 'expect', 'tags'} for case in CASES)
    assert all(case['prompt'].strip() and case['expect'].strip() and case['tags'] for case in CASES)


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['id'])
def test_each_acceptance_case_has_an_executable_contract(case):
    case_id = case['id']
    prompt = case['prompt']
    if case_id == 'G01':
        assert route_request([{'role': 'user', 'content': prompt}]).route == 'assistant_meta'
    elif case_id == 'G02':
        assert route_request([{'role': 'user', 'content': prompt}]).route == 'greeting_help'
    elif case_id in {'G04', 'G05', 'G06', 'G07', 'G08'}:
        assert route_request([{'role': 'user', 'content': prompt}]).route == 'confluence_grounded'
    elif case_id == 'G09':
        assert route_request([{'role': 'user', 'content': prompt}]).route == 'out_of_scope'
    elif case_id == 'G10':
        assert route_request([{'role': 'user', 'content': prompt}]).route == 'out_of_scope'
    elif case_id == 'S05':
        assert authorize_tool('acceptance', 'unknown', {}).reason == 'unknown_tool'
    elif case_id == 'S06':
        assert (
            authorize_tool('acceptance', 'search_confluence', {'query': 'x', 'extra': 1}).reason == 'invalid_arguments'
        )
    elif case_id.startswith('D'):
        extension = {
            '1': 'pdf',
            '2': 'docx',
            '3': 'xlsx',
            '4': 'pptx',
            '5': 'pdf',
            '6': 'docx',
            '7': 'xlsx',
            '8': 'pptx',
        }[case_id[-1]]
        allowed = HERMES_ALLOWED_ATTACHMENT_EXTENSIONS if int(case_id[-1]) <= 4 else HERMES_ALLOWED_ARTIFACT_EXTENSIONS
        assert extension in allowed
    else:
        assert case_id in LIVE_IDS
        assert any(
            tag in case['tags']
            for tag in {'cancellation', 'two-user', 'security', 'memory', 'self-development', 'load'}
        )


def test_live_acceptance_cases_are_explicitly_separated():
    assert LIVE_IDS < EXPECTED_IDS
    assert {'L01', 'L02', 'P02', 'P03', 'S01', 'S02'} <= LIVE_IDS
