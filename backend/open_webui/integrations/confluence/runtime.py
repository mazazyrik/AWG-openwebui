"""Server-only response contract for AWG GPT requests."""

from __future__ import annotations

import codecs
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from fastapi.responses import JSONResponse
from open_webui.utils.json_codec import JSONCodec
from open_webui.utils.misc import (
    get_output_text,
    openai_chat_chunk_message_template,
    openai_chat_completion_message_template,
)
from starlette.responses import StreamingResponse

STATE_KEY = 'awg_confluence_grounding'
ATTESTATION_KEY = 'awg_confluence_attestations'
INVOCATION_KEY = 'awg_confluence_invocations'
STATE_VERSION = 2
MAX_PROVIDER_RESPONSE_BYTES = 65_536
MAX_PROVIDER_RESPONSE_CHARS = 16_384

Route = Literal[
    'assistant_meta',
    'memory_command',
    'greeting_help',
    'corporate_profile',
    'confluence_grounded',
    'clarification',
    'out_of_scope',
]


@dataclass(frozen=True)
class AwgRequestState:
    state_version: int
    route: Route
    model_id: str
    invocation_id: str
    filter_id: str
    profile_version: str
    prompt_hash: str
    sources: tuple[dict, ...]
    memory_operation: str | None
    scope_decision: str
    unavailable: bool
    unavailable_reason: str | None
    client_stream: bool
    provider_required: bool
    deterministic_answer: str | None


@dataclass
class AwgRequestRegistry:
    states: dict[tuple[str, str], AwgRequestState | None]


@dataclass(frozen=True)
class AwgAttachmentAttestation:
    model_id: str
    invocation_id: str
    filter_id: str
    client_stream: bool


@dataclass
class AwgAttestationRegistry:
    attached: dict[tuple[str, str, str], AwgAttachmentAttestation]


@dataclass(frozen=True)
class AwgInvocation:
    model_id: str
    invocation_id: str


@dataclass
class AwgInvocationRegistry:
    invocations: dict[int, AwgInvocation]


class AwgResponseRejected(ValueError):
    pass


def register_awg_invocation(request, model: dict, metadata: dict) -> str:
    """Create a server-only invocation correlation id for one model task."""
    registry = getattr(request.state, INVOCATION_KEY, None)
    if not isinstance(registry, AwgInvocationRegistry):
        registry = AwgInvocationRegistry(invocations={})
        setattr(request.state, INVOCATION_KEY, registry)
    model_id = str(model.get('id') or '') if isinstance(model, dict) else ''
    invocation_id = str(uuid4())
    registry.invocations[id(metadata)] = AwgInvocation(model_id, invocation_id)
    return invocation_id


def get_awg_invocation_id(request, model: dict, metadata: dict | None) -> str:
    """Return a server-only invocation id bound to this metadata object."""
    registry = getattr(request.state, INVOCATION_KEY, None)
    if not isinstance(registry, AwgInvocationRegistry) or not isinstance(metadata, dict):
        return ''
    model_id = str(model.get('id') or '') if isinstance(model, dict) else ''
    registered = registry.invocations.get(id(metadata))
    return registered.invocation_id if registered and registered.model_id == model_id else ''


def attest_awg_attachment(
    request,
    model: dict,
    metadata: dict | None,
    filter_id: str,
    client_stream: bool,
) -> str:
    """Attest an attached AWG filter independently from its mutable state."""
    model_id = str(model.get('id') or '') if isinstance(model, dict) else ''
    invocation_id = get_awg_invocation_id(request, model, metadata)
    if not model_id or not invocation_id or not filter_id:
        return ''
    registry = getattr(request.state, ATTESTATION_KEY, None)
    if not isinstance(registry, AwgAttestationRegistry):
        registry = AwgAttestationRegistry(attached={})
        setattr(request.state, ATTESTATION_KEY, registry)
    key = (model_id, invocation_id, filter_id)
    registry.attached[key] = AwgAttachmentAttestation(
        model_id=model_id,
        invocation_id=invocation_id,
        filter_id=filter_id,
        client_stream=client_stream,
    )
    return invocation_id


def get_awg_client_stream(request, model: dict, metadata: dict | None) -> bool:
    """Return the original response contract from server attachment attestation."""
    model_id = str(model.get('id') or '') if isinstance(model, dict) else ''
    invocation_id = get_awg_invocation_id(request, model, metadata)
    registry = getattr(request.state, ATTESTATION_KEY, None)
    if not isinstance(registry, AwgAttestationRegistry):
        return False
    contracts = {
        attestation.client_stream
        for attestation in registry.attached.values()
        if attestation.model_id == model_id and attestation.invocation_id == invocation_id
    }
    return contracts == {True}


def set_awg_request_state(
    request,
    model_id: str,
    invocation_id: str,
    state: AwgRequestState | None,
) -> None:
    """Store AWG state without leaking it across side-by-side models."""
    registry = getattr(request.state, STATE_KEY, None)
    if not isinstance(registry, AwgRequestRegistry):
        registry = AwgRequestRegistry(states={})
        setattr(request.state, STATE_KEY, registry)
    registry.states[(model_id, invocation_id)] = state


def get_awg_request_state(
    request,
    model: dict,
    metadata: dict | None,
) -> tuple[bool, AwgRequestState | None]:
    """Return a validated state created by the attached AWG filter."""
    model_id = str(model.get('id') or '') if isinstance(model, dict) else ''
    invocation_id = get_awg_invocation_id(request, model, metadata)
    attestations = getattr(request.state, ATTESTATION_KEY, None)
    attached_filter_ids = (
        {
            attestation.filter_id
            for attestation in attestations.attached.values()
            if attestation.model_id == model_id and attestation.invocation_id == invocation_id
        }
        if isinstance(attestations, AwgAttestationRegistry)
        else set()
    )
    if not invocation_id or not attached_filter_ids:
        return False, None
    registry = getattr(request.state, STATE_KEY, None)
    if not isinstance(registry, AwgRequestRegistry):
        return True, None
    key = (model_id, invocation_id)
    if key not in registry.states:
        return True, None
    state = registry.states[key]
    if not isinstance(state, AwgRequestState) or state.state_version != STATE_VERSION:
        return True, None

    filter_ids = ((model.get('info') or {}).get('meta') or {}).get('filterIds') or []
    if (
        state.model_id != model_id
        or state.invocation_id != invocation_id
        or state.filter_id not in attached_filter_ids
        or state.filter_id not in filter_ids
    ):
        return True, None
    if state.provider_required != (state.route == 'confluence_grounded'):
        return True, None
    if state.provider_required == (state.deterministic_answer is not None):
        return True, None
    return True, state


def clear_awg_request_state(request, model: dict, metadata: dict | None) -> None:
    """Delete one invocation state without touching neighboring model tasks."""
    model_id = str(model.get('id') or '') if isinstance(model, dict) else ''
    invocation_id = get_awg_invocation_id(request, model, metadata)
    registry = getattr(request.state, STATE_KEY, None)
    if isinstance(registry, AwgRequestRegistry):
        registry.states.pop((model_id, invocation_id), None)
    if isinstance(registry, AwgRequestRegistry) and not registry.states:
        delattr(request.state, STATE_KEY)

    attestations = getattr(request.state, ATTESTATION_KEY, None)
    if isinstance(attestations, AwgAttestationRegistry):
        attestations.attached = {
            key: attestation
            for key, attestation in attestations.attached.items()
            if (attestation.model_id, attestation.invocation_id) != (model_id, invocation_id)
        }
        if not attestations.attached:
            delattr(request.state, ATTESTATION_KEY)

    invocations = getattr(request.state, INVOCATION_KEY, None)
    if isinstance(invocations, AwgInvocationRegistry) and isinstance(metadata, dict):
        invocations.invocations.pop(id(metadata), None)
        if not invocations.invocations:
            delattr(request.state, INVOCATION_KEY)


def _extract_response_text(payload: dict) -> str:
    choices = payload.get('choices')
    choice_text = None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get('message')
        if isinstance(message, dict):
            if message.get('tool_calls') or message.get('reasoning_content') or message.get('reasoning'):
                raise AwgResponseRejected('provider_response_contains_non_text_output')
            choice_text = message.get('content')
            if choice_text is not None and not isinstance(choice_text, str):
                raise AwgResponseRejected('provider_response_content_invalid')

    output = payload.get('output')
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict) or item.get('type') != 'message':
                raise AwgResponseRejected('provider_response_output_invalid')
            content = item.get('content')
            if not isinstance(content, list) or any(
                not isinstance(part, dict) or part.get('type') != 'output_text' or not isinstance(part.get('text'), str)
                for part in content
            ):
                raise AwgResponseRejected('provider_response_output_invalid')
    output_text = get_output_text(output) if isinstance(output, list) else None
    if choice_text is not None and output_text and choice_text != output_text:
        raise AwgResponseRejected('provider_response_shapes_mismatch')
    text = choice_text if choice_text is not None else output_text
    if not isinstance(text, str):
        raise AwgResponseRejected('provider_response_text_missing')
    if len(text) > MAX_PROVIDER_RESPONSE_CHARS:
        raise AwgResponseRejected('provider_response_too_large')
    return text


def _json_response_payload(response: object) -> dict | None:
    if isinstance(response, dict):
        try:
            size = len(JSONCodec.dumps(response).encode('utf-8'))
        except (TypeError, ValueError) as error:
            raise AwgResponseRejected('provider_response_json_invalid') from error
        if size > MAX_PROVIDER_RESPONSE_BYTES:
            raise AwgResponseRejected('provider_response_too_large')
        return response
    if isinstance(response, JSONResponse):
        raw = response.body
        if not isinstance(raw, bytes) or len(raw) > MAX_PROVIDER_RESPONSE_BYTES:
            raise AwgResponseRejected('provider_response_too_large')
        try:
            payload = JSONCodec.loads(raw.decode('utf-8', 'strict'))
        except (UnicodeDecodeError, JSONCodec.JSONDecodeError) as error:
            raise AwgResponseRejected('provider_response_json_invalid') from error
        return payload if isinstance(payload, dict) else None
    return None


def _stream_event_text(value: str) -> list[str]:
    try:
        event = JSONCodec.loads(value)
    except JSONCodec.JSONDecodeError as error:
        raise AwgResponseRejected('provider_stream_json_invalid') from error
    if not isinstance(event, dict):
        raise AwgResponseRejected('provider_stream_event_invalid')
    if event.get('type') == 'response.output_text.delta':
        delta = event.get('delta')
        if not isinstance(delta, str):
            raise AwgResponseRejected('provider_stream_delta_invalid')
        return [delta]

    texts = []
    choices = event.get('choices')
    if not isinstance(choices, list):
        return texts
    for choice in choices:
        delta = choice.get('delta') if isinstance(choice, dict) else None
        if not isinstance(delta, dict):
            continue
        if delta.get('tool_calls') or delta.get('reasoning_content') or delta.get('reasoning'):
            raise AwgResponseRejected('provider_stream_contains_non_text_output')
        content = delta.get('content')
        if content is not None:
            if not isinstance(content, str):
                raise AwgResponseRejected('provider_stream_content_invalid')
            texts.append(content)
    return texts


async def extract_awg_provider_text(response: object) -> str:
    """Buffer a provider response and return bounded assistant text."""
    payload = _json_response_payload(response)
    if payload is not None:
        return _extract_response_text(payload)
    if not isinstance(response, StreamingResponse):
        raise AwgResponseRejected('provider_response_type_invalid')

    raw_size = 0
    text_parts: list[str] = []
    pending = ''
    decoder = codecs.getincrementaldecoder('utf-8')('strict')
    try:
        async for chunk in response.body_iterator:
            raw = chunk if isinstance(chunk, bytes) else str(chunk).encode('utf-8')
            raw_size += len(raw)
            if raw_size > MAX_PROVIDER_RESPONSE_BYTES:
                raise AwgResponseRejected('provider_stream_too_large')
            try:
                pending += decoder.decode(raw, final=False)
            except UnicodeDecodeError as error:
                raise AwgResponseRejected('provider_stream_utf8_invalid') from error
            lines = pending.splitlines(keepends=True)
            pending = ''
            if lines and not lines[-1].endswith(('\n', '\r')):
                pending = lines.pop()
            for part in lines:
                value = part.removeprefix('data:').strip()
                if not value or value == '[DONE]':
                    continue
                text_parts.extend(_stream_event_text(value))
                if sum(len(part) for part in text_parts) > MAX_PROVIDER_RESPONSE_CHARS:
                    raise AwgResponseRejected('provider_stream_text_too_large')
        try:
            pending += decoder.decode(b'', final=True)
        except UnicodeDecodeError as error:
            raise AwgResponseRejected('provider_stream_utf8_invalid') from error
        if pending.strip():
            value = pending.removeprefix('data:').strip()
            if value and value != '[DONE]':
                text_parts.extend(_stream_event_text(value))
                if sum(len(part) for part in text_parts) > MAX_PROVIDER_RESPONSE_CHARS:
                    raise AwgResponseRejected('provider_stream_text_too_large')
    finally:
        if hasattr(response.body_iterator, 'aclose'):
            await response.body_iterator.aclose()
        if response.background is not None:
            await response.background()
    if not text_parts:
        raise AwgResponseRejected('provider_stream_text_missing')
    return ''.join(text_parts)


def build_awg_response(answer: str, model_id: str, stream: bool) -> dict | StreamingResponse:
    """Build a canonical provider-shaped response containing only the safe answer."""
    if stream:

        async def stream_content():
            message = openai_chat_chunk_message_template(model_id, answer)
            yield f'data: {JSONCodec.dumps(message)}\n\n'
            finish = openai_chat_chunk_message_template(model_id, '')
            finish['choices'][0]['finish_reason'] = 'stop'
            yield f'data: {JSONCodec.dumps(finish)}\n\n'
            yield 'data: [DONE]\n\n'

        return StreamingResponse(stream_content(), media_type='text/event-stream')
    return openai_chat_completion_message_template(model_id, answer)
