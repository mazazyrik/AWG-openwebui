import pytest

from open_webui.utils.payload import apply_model_params_to_body_ollama, remove_open_webui_params
from open_webui.utils.tool_call_limits import _resolve_profile_max_tool_call_iterations


@pytest.mark.parametrize(
    ('value', 'expected'),
    [(1, 1), (4, 4), (8, 8), (0, 1), (-2, 1), (9, 8)],
)
def test_profile_tool_call_limit_is_bounded(value: int, expected: int) -> None:
    assert _resolve_profile_max_tool_call_iterations(value, has_explicit_override=False) == expected


@pytest.mark.parametrize('value', [True, False, 2.5, '4', None])
def test_invalid_profile_tool_call_limit_is_ignored(value: object) -> None:
    assert _resolve_profile_max_tool_call_iterations(value, has_explicit_override=False) is None


def test_explicit_request_state_limit_is_preserved() -> None:
    assert _resolve_profile_max_tool_call_iterations(4, has_explicit_override=True) is None


def test_open_webui_tool_call_limit_is_removed_from_provider_params() -> None:
    params = {'max_tool_call_iterations': 4, 'temperature': 0.2}

    assert remove_open_webui_params(params) == {'temperature': 0.2}


def test_ollama_body_does_not_receive_tool_call_limit() -> None:
    form_data = {'options': {}}

    result = apply_model_params_to_body_ollama({'max_tool_call_iterations': 4, 'temperature': 0.2}, form_data)

    assert result['options'] == {'temperature': 0.2}
