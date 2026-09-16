"""Validated identity profile and system prompt for AWG GPT."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PROFILE_PATH = Path(__file__).with_name('awg_profile.json')
PROMPT_PATH = Path(__file__).with_name('manager_prompt.md')
MAX_PROFILE_BYTES = 32_768
MAX_PROMPT_BYTES = 32_768
PROFILE_SCHEMA_VERSION = 1
PROMPT_MARKER = 'AWG_GPT_POLICY_BEGIN'
RESPONSE_KEYS = {
    'assistant_meta',
    'greeting_help',
    'corporate_profile_unavailable',
    'memory_add',
    'memory_remove',
    'memory_list',
    'memory_failure',
    'out_of_scope',
    'clarification',
}
TEMPLATE_FIELDS = {
    'ASSISTANT_NAME',
    'COMPANY_NAME',
    'IDENTITY_VERSION',
    'ROLE',
    'ALIASES',
    'APPROVED_CONTEXT',
    'SUPPORTED_SCOPE',
    'OUT_OF_SCOPE_RESPONSE',
}
INSTRUCTION_PATTERNS = (
    re.compile(
        r'(?:ignore|disregard|override).{0,40}(?:instructions?|rules?|prompt)',
        re.IGNORECASE,
    ),
    re.compile(
        r'(?:игнорируй|забудь|переопредели|смени).{0,40}(?:инструкц\w*|правил\w*|роль|промпт)',
        re.IGNORECASE,
    ),
    re.compile(r'<\/?(?:system|assistant|user)>', re.IGNORECASE),
    re.compile(r'\b(?:SOURCE_DATA_JSON|FINAL_ROUTE|AWG_GPT_POLICY)\b', re.IGNORECASE),
)


@dataclass(frozen=True)
class CorporateFact:
    statement: str
    source_url: str
    owner: str
    as_of: str


@dataclass(frozen=True)
class RetrievalAlias:
    alias: str
    query: str
    owner: str
    as_of: str
    provenance: str


@dataclass(frozen=True)
class AwgProfile:
    schema_version: int
    identity_version: str
    assistant_name: str
    aliases: tuple[str, ...]
    role: str
    company_name: str
    approved_context: tuple[CorporateFact, ...]
    retrieval_aliases: tuple[RetrievalAlias, ...]
    supported_scope: tuple[str, ...]
    responses: dict[str, str]


def _require_keys(value: dict[str, Any], expected: set[str], location: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f'Invalid AWG profile keys at {location}: missing={missing}, extra={extra}')


def _require_mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f'AWG profile field {location} must be an object')
    return value


def _require_text(value: object, location: str, *, max_chars: int = 2000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'AWG profile field {location} must be a non-empty string')
    text = value.strip()
    if len(text) > max_chars or '\x00' in text:
        raise ValueError(f'AWG profile field {location} is invalid')
    if any(pattern.search(text) for pattern in INSTRUCTION_PATTERNS):
        raise ValueError(f'AWG profile field {location} contains policy-like instructions')
    return text


def _require_text_list(value: object, location: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f'AWG profile field {location} must be a list of strings')
    items = tuple(_require_text(item, f'{location}[{index}]', max_chars=500) for index, item in enumerate(value))
    if len(items) != len(set(item.casefold() for item in items)):
        raise ValueError(f'AWG profile field {location} contains duplicates')
    return items


def _require_date(value: object, location: str) -> str:
    text = _require_text(value, location, max_chars=10)
    try:
        dt.date.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f'AWG profile field {location} contains an invalid date') from error
    return text


def _load_corporate_facts(value: object) -> tuple[CorporateFact, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError('AWG profile company.approved_context must contain approved facts')
    facts = []
    for index, raw in enumerate(value):
        location = f'company.approved_context[{index}]'
        item = _require_mapping(raw, location)
        _require_keys(item, {'statement', 'source_url', 'owner', 'as_of'}, location)
        source_url = _require_text(item['source_url'], f'{location}.source_url', max_chars=300)
        parsed = urlsplit(source_url)
        if parsed.scheme != 'https' or parsed.netloc != 'www.awg.ru':
            raise ValueError(f'AWG profile field {location}.source_url must use https://www.awg.ru')
        facts.append(
            CorporateFact(
                statement=_require_text(item['statement'], f'{location}.statement', max_chars=600),
                source_url=source_url,
                owner=_require_text(item['owner'], f'{location}.owner', max_chars=100),
                as_of=_require_date(item['as_of'], f'{location}.as_of'),
            )
        )
    return tuple(facts)


def _load_retrieval_aliases(value: object) -> tuple[RetrievalAlias, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError('AWG profile retrieval_aliases must contain approved aliases')
    aliases = []
    for index, raw in enumerate(value):
        location = f'retrieval_aliases[{index}]'
        item = _require_mapping(raw, location)
        _require_keys(item, {'alias', 'query', 'owner', 'as_of', 'provenance'}, location)
        aliases.append(
            RetrievalAlias(
                alias=_require_text(item['alias'], f'{location}.alias', max_chars=80),
                query=_require_text(item['query'], f'{location}.query', max_chars=300),
                owner=_require_text(item['owner'], f'{location}.owner', max_chars=100),
                as_of=_require_date(item['as_of'], f'{location}.as_of'),
                provenance=_require_text(item['provenance'], f'{location}.provenance', max_chars=200),
            )
        )
    if len(aliases) != len({item.alias.casefold() for item in aliases}):
        raise ValueError('AWG profile retrieval_aliases contains duplicate aliases')
    return tuple(aliases)


def load_awg_profile(path: Path = PROFILE_PATH) -> AwgProfile:
    """Load and validate the repository-owned AWG identity profile."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f'Unable to read AWG profile: {path}') from error
    if not raw or len(raw) > MAX_PROFILE_BYTES:
        raise ValueError('AWG profile size is invalid')
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('AWG profile is not valid UTF-8 JSON') from error

    root = _require_mapping(payload, 'root')
    _require_keys(
        root,
        {'schema_version', 'identity_version', 'assistant', 'company', 'retrieval_aliases', 'scope', 'responses'},
        'root',
    )
    if type(root['schema_version']) is not int or root['schema_version'] != PROFILE_SCHEMA_VERSION:
        raise ValueError(f'Unsupported AWG profile schema version: {root["schema_version"]!r}')

    identity_version = _require_text(root['identity_version'], 'identity_version', max_chars=40)
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}\.\d+', identity_version):
        raise ValueError('AWG profile identity_version must use YYYY-MM-DD.N format')
    try:
        dt.date.fromisoformat(identity_version.split('.', 1)[0])
    except ValueError as error:
        raise ValueError('AWG profile identity_version contains an invalid date') from error

    assistant = _require_mapping(root['assistant'], 'assistant')
    _require_keys(assistant, {'name', 'aliases', 'role'}, 'assistant')
    company = _require_mapping(root['company'], 'company')
    _require_keys(company, {'name', 'approved_context'}, 'company')
    scope = _require_mapping(root['scope'], 'scope')
    _require_keys(scope, {'strict', 'supported'}, 'scope')
    if scope['strict'] is not True:
        raise ValueError('AWG profile scope.strict must be true')
    responses = _require_mapping(root['responses'], 'responses')
    _require_keys(responses, RESPONSE_KEYS, 'responses')

    name = _require_text(assistant['name'], 'assistant.name', max_chars=80)
    aliases = _require_text_list(assistant['aliases'], 'assistant.aliases')
    alias_keys = {alias.casefold() for alias in aliases}
    if name != 'AWG GPT' or not {'avg', 'авг'} <= alias_keys:
        raise ValueError('AWG profile must use AWG GPT and include Latin and Cyrillic typo aliases')
    company_name = _require_text(company['name'], 'company.name', max_chars=80)
    if company_name != 'AWG':
        raise ValueError('AWG profile company.name must be AWG')

    return AwgProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        identity_version=identity_version,
        assistant_name=name,
        aliases=aliases,
        role=_require_text(assistant['role'], 'assistant.role', max_chars=300),
        company_name=company_name,
        approved_context=_load_corporate_facts(company['approved_context']),
        retrieval_aliases=_load_retrieval_aliases(root['retrieval_aliases']),
        supported_scope=_require_text_list(scope['supported'], 'scope.supported'),
        responses={key: _require_text(responses[key], f'responses.{key}') for key in RESPONSE_KEYS},
    )


def render_system_prompt(profile: AwgProfile, path: Path = PROMPT_PATH) -> str:
    """Render the trusted system prompt from the validated profile."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f'Unable to read AWG prompt template: {path}') from error
    if not raw or len(raw) > MAX_PROMPT_BYTES:
        raise ValueError('AWG prompt template size is invalid')
    try:
        template = raw.decode('utf-8')
    except UnicodeDecodeError as error:
        raise ValueError('AWG prompt template is not valid UTF-8') from error

    fields = set(re.findall(r'\{\{([A-Z_]+)}}', template))
    if fields != TEMPLATE_FIELDS:
        raise ValueError(
            f'Invalid AWG prompt template fields: expected={sorted(TEMPLATE_FIELDS)}, actual={sorted(fields)}'
        )
    values = {
        'ASSISTANT_NAME': profile.assistant_name,
        'COMPANY_NAME': profile.company_name,
        'IDENTITY_VERSION': profile.identity_version,
        'ROLE': profile.role,
        'ALIASES': ', '.join(profile.aliases),
        'APPROVED_CONTEXT': (
            '\n'.join(
                f'- {fact.statement} Источник: {fact.source_url}; владелец: {fact.owner}; актуально на {fact.as_of}.'
                for fact in profile.approved_context
            )
        ),
        'SUPPORTED_SCOPE': '\n'.join(f'- {item}' for item in profile.supported_scope),
        'OUT_OF_SCOPE_RESPONSE': profile.responses['out_of_scope'],
    }
    prompt = template
    for key, value in values.items():
        prompt = prompt.replace(f'{{{{{key}}}}}', value)
    if '{{' in prompt or '}}' in prompt:
        raise ValueError('AWG prompt template contains unresolved fields')
    return prompt.strip()


def prompt_sha256(prompt: str) -> str:
    """Return a stable digest for runtime diagnostics."""
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()
