from typing import Any


def _resolve_profile_max_tool_call_iterations(value: Any, *, has_explicit_override: bool) -> int | None:
    if has_explicit_override or not isinstance(value, int) or isinstance(value, bool):
        return None

    return min(8, max(1, value))
