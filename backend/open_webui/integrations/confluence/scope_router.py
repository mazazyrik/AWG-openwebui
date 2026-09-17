"""Deterministic request routing for the AWG GPT filter."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Route = Literal[
    'assistant_meta',
    'memory_command',
    'greeting_help',
    'corporate_profile',
    'confluence_grounded',
    'clarification',
    'out_of_scope',
]
MemoryOperation = Literal['add', 'remove', 'list']
MemoryKind = Literal['alias', 'preference']

AWG_MARKER_RE = re.compile(r'\b(?:awg|avg|авг)(?:\s+gpt)?\b', re.IGNORECASE)
UNANCHORED_PRONOUN_RE = re.compile(
    r'\b(?:у\s+них|они|их|там|them|their|they|there)\b',
    re.IGNORECASE,
)
CORPORATE_POSSESSIVE_RE = re.compile(
    r'\b(?:у\s+нас|наш(?:а|и|е|его|ей|ему|им|их)?|our|ours)\b',
    re.IGNORECASE,
)
CORPORATE_FIRST_PERSON_RE = re.compile(r'\b(?:мы|we)\b', re.IGNORECASE)
UNRELATED_COMPANY_RE = re.compile(r'\b(?:яндекс\w*|yandex)\b', re.IGNORECASE)
GREETING_RE = re.compile(
    r'^\s*(?:(?:привет|здравствуй(?:те)?|доброе\s+'
    r'(?:утро|день|вечер)|hi|hello|hey)[!,.\s]*)+'
    r'(?:кто\s+ты|что(?:\s+ты)?\s+умеешь|помоги|help)?[!?.\s]*$',
    re.IGNORECASE,
)
META_RE = re.compile(
    r'\b(?:кто\s+ты|что\s+ты\s+такое|как\s+тебя\s+зовут|представься|'
    r'что(?:\s+ты)?\s+(?:умеешь|можешь)|чем\s+ты\s+можешь\s+помочь|твоя\s+(?:роль|задача)|'
    r'какие\s+у\s+тебя\s+возможности|расскажи\s+о\s+себе|какой\s+ты\s+ассистент|'
    r'для\s+чего\s+ты\s+нужен|who\s+are\s+you|what\s+are\s+you|what(?:\s+is|\'s)\s+your\s+name|'
    r'tell\s+me\s+about\s+yourself|'
    r'what\s+can\s+you\s+do|how\s+can\s+you\s+help|what\s+is\s+your\s+(?:role|purpose))\b',
    re.IGNORECASE,
)
PROFILE_RE = re.compile(
    r'\b(?:кто\s+(?:такие\s+)?мы|что\s+мы\s+делаем|чем\s+мы\s+занимаемся|'
    r'who\s+are\s+we|what\s+do\s+we\s+do|'
    r'что\s+такое\s+(?:awg|avg|авг)|чем\s+занимается\s+(?:компания\s+)?(?:awg|avg|авг)|'
    r'расскажи\s+(?:мне\s+)?(?:о|про)\s+(?:компанию\s+)?(?:awg|avg|авг)|'
    r'(?:какие\s+)?услуг\w*\s+(?:(?:у|компании)\s+)?(?:awg|avg|авг)|'
    r'экспертиз\w*\s+(?:awg|avg|авг)|what\s+does\s+(?:awg|avg|авг)\s+do|'
    r'(?:awg|avg|авг)\s+services|about\s+(?:awg|avg|авг))\b',
    re.IGNORECASE,
)
MEMORY_ADD_RE = re.compile(
    r'\b(?:запомни|сохрани\s+(?:в\s+)?памят|remember\s+that|save\s+(?:this|to\s+memory))\b',
    re.IGNORECASE,
)
MEMORY_REMOVE_RE = re.compile(
    r'\b(?:забудь|удали\s+(?:из\s+)?памят|forget\s+that|remove\s+from\s+memory)\b',
    re.IGNORECASE,
)
MEMORY_LIST_RE = re.compile(
    r'\b(?:что\s+ты\s+(?:обо\s+мне\s+)?помнишь|покажи\s+(?:мою\s+)?память|'
    r'what\s+do\s+you\s+remember|show\s+(?:my\s+)?memor(?:y|ies))\b',
    re.IGNORECASE,
)
GROUNDED_INTENT_RE = re.compile(
    r'\b(?:проект\w*|команд\w*|клиент\w*|сотрудник\w*|разработчик\w*|'
    r'менеджер\w*|документ\w*|инструкц\w*|процесс\w*|регламент\w*|'
    r'встреч\w*|решени\w*|статус\w*|согласов\w*|отпуск\w*|офис\w*|'
    r'заказчик\w*|договор\w*|ваканси\w*|должност\w*|отдел\w*|'
    r'projects?|teams?|clients?|employees?|developers?|managers?|documents?|process(?:es)?|'
    r'polic(?:y|ies)|meetings?|status(?:es)?|confluence)\b',
    re.IGNORECASE,
)
DELIVERY_QUESTION_RE = re.compile(
    r'\b(?:какую|какие|что|чем|what|which)\b.{0,80}'
    r'\b(?:разработк\w*|внедрен\w*|интеграц\w*|геймификац\w*|релиз\w*|'
    r'development|implementation|integration|gamification|releases?)\b.{0,80}'
    r'\b(?:дела\w*|разрабатыва\w*|внедря\w*|интегрир\w*|реализу\w*|выпуска\w*|'
    r'develop\w*|implement\w*|integrat\w*|deliver\w*|releas\w*)\b',
    re.IGNORECASE,
)
KRATNO_DELIVERY_QUESTION_RE = re.compile(
    r'^\s*какую\s+разработку\s+по\s+геймификации\s+мы\s+делали\s*[?!.]*\s*$',
    re.IGNORECASE,
)
KRATNO_ALIAS_RE = re.compile(r'\b(?:кратно|servity)\b', re.IGNORECASE)
OTHER_PROJECT_RE = re.compile(r'\b(?:спортмастер\w*|sportmaster\w*|яндекс\w*|yandex)\b', re.IGNORECASE)
POLICY_ATTACK_RE = re.compile(
    r'\b(?:покажи|раскрой|выведи|повтори|пришли|show|reveal|print|repeat)\b.{0,40}'
    r'\b(?:системн\w*\s+(?:промпт|инструкц\w*)|system\s+prompt|'
    r'hidden\s+instructions?|токен\w*|secrets?)\b|'
    r'\b(?:ignore\s+(?:all\s+)?previous\s+instructions?|'
    r'игнорируй\s+(?:все\s+)?предыдущие\s+инструкции)\b',
    re.IGNORECASE,
)
ALIAS_ADD_PATTERNS = (
    re.compile(
        r'(?:запомни|сохрани)(?:,|\s)+(?:что\s+)?(?P<key>[^,.;:\n]{1,80}?)\s+'
        r'(?:означает|значит|это)\s+(?P<value>[^\n]{1,300})',
        re.IGNORECASE,
    ),
    re.compile(
        r'(?:запомни|сохрани)(?:,|\s)+(?:что\s+)?под\s+(?P<key>[^,.;:\n]{1,80}?)\s+'
        r'(?:я\s+)?(?:имею\s+в\s+виду|понимаю)\s+(?P<value>[^\n]{1,300})',
        re.IGNORECASE,
    ),
    re.compile(
        r'(?:remember\s+that|save(?:\s+to\s+memory)?)[,:\s]+'
        r'(?P<key>[^,.;:\n]{1,80}?)\s+(?:means|refers\s+to|is)\s+(?P<value>[^\n]{1,300})',
        re.IGNORECASE,
    ),
)
PREFERENCE_ADD_RE = re.compile(
    r'(?:запомни|сохрани)(?:,|\s)+(?:что\s+)?(?:я\s+)?'
    r'(?:предпочитаю|хочу\s+получать\s+ответы)\s+(?P<value>[^\n]{1,300})|'
    r'(?:remember\s+that|save(?:\s+to\s+memory)?)[,:\s]+(?:i\s+)?prefer\s+(?P<value_en>[^\n]{1,300})',
    re.IGNORECASE,
)
REMOVE_VALUE_RE = re.compile(
    r'(?:забудь|удали\s+(?:из\s+)?памяти)(?:,|\s)+(?:что\s+)?(?P<value>[^\n]{1,300})|'
    r'(?:forget\s+that|remove\s+from\s+memory)[,:\s]+(?P<value_en>[^\n]{1,300})',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MemoryCommand:
    operation: MemoryOperation
    kind: MemoryKind | None = None
    key: str | None = None
    value: str | None = None


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    scope_decision: str
    memory_operation: MemoryOperation | None = None


def latest_user_text(messages: list[dict]) -> str:
    """Return the latest bounded textual user message."""
    return next(
        (
            message['content'].strip()[:2000]
            for message in reversed(messages)
            if message.get('role') == 'user'
            and isinstance(message.get('content'), str)
            and message['content'].strip()
        ),
        '',
    )


def _contains_alias(text: str, aliases: tuple[str, ...]) -> bool:
    return any(re.search(rf'(?<!\w){re.escape(alias)}(?!\w)', text, re.IGNORECASE) for alias in aliases)


def parse_memory_command(question: str) -> MemoryCommand | None:
    """Parse bounded explicit alias and preference memory commands."""
    if MEMORY_LIST_RE.search(question):
        return MemoryCommand('list')
    if MEMORY_REMOVE_RE.search(question):
        match = REMOVE_VALUE_RE.search(question)
        value = (match.group('value') or match.group('value_en')).strip(' .!?') if match else None
        return MemoryCommand('remove', value=value)
    if not MEMORY_ADD_RE.search(question):
        return None
    for pattern in ALIAS_ADD_PATTERNS:
        match = pattern.search(question)
        if match:
            return MemoryCommand(
                'add',
                'alias',
                match.group('key').strip(' «»"\''),
                match.group('value').strip(' .!?'),
            )
    match = PREFERENCE_ADD_RE.search(question)
    if match:
        value = match.group('value') or match.group('value_en')
        return MemoryCommand('add', 'preference', value=value.strip(' .!?'))
    return MemoryCommand('add')


def _has_prior_awg_anchor(messages: list[dict]) -> bool:
    user_messages = [
        message['content']
        for message in messages
        if message.get('role') == 'user' and isinstance(message.get('content'), str)
    ][-5:-1]
    return any(AWG_MARKER_RE.search(text) for text in user_messages)


def _has_prior_kratno_anchor(messages: list[dict]) -> bool:
    user_messages = [
        message['content']
        for message in messages
        if message.get('role') == 'user' and isinstance(message.get('content'), str)
    ]
    if len(user_messages) < 2:
        return False
    previous_question = user_messages[-2]
    return bool(
        KRATNO_ALIAS_RE.search(previous_question)
        or KRATNO_DELIVERY_QUESTION_RE.fullmatch(previous_question)
    )


def _is_kratno_follow_up(messages: list[dict], question: str) -> bool:
    return not OTHER_PROJECT_RE.search(question) and _has_prior_kratno_anchor(messages) and bool(
        GROUNDED_INTENT_RE.search(question) or DELIVERY_QUESTION_RE.search(question)
    )


def _kratno_scope_decision(messages: list[dict], question: str) -> str | None:
    if KRATNO_ALIAS_RE.search(question) or KRATNO_DELIVERY_QUESTION_RE.fullmatch(question):
        return 'awg_kratno_delivery_question'
    if _is_kratno_follow_up(messages, question):
        return 'awg_kratno_delivery_question'
    return None


def needs_project_clarification(
    messages: list[dict],
    approved_aliases: tuple[str, ...] = (),
    personal_alias: bool = False,
) -> bool:
    """Require an AWG anchor for unresolved third-person references."""
    question = latest_user_text(messages)
    if not question:
        return False
    if AWG_MARKER_RE.search(question) or _contains_alias(question, approved_aliases) or personal_alias:
        return False
    if _is_kratno_follow_up(messages, question):
        return False
    prior_anchor = _has_prior_awg_anchor(messages)
    if UNANCHORED_PRONOUN_RE.search(question):
        return not prior_anchor
    return not prior_anchor and bool(
        re.fullmatch(
            r'(?:а\s+)?кто\s+(?:главный|в команде|разработчик|менеджер)\s*[?!.]*',
            question,
            re.IGNORECASE,
        )
    )


def _grounded_scope_decision(
    messages: list[dict],
    question: str,
    approved_aliases: tuple[str, ...],
    personal_alias: bool,
) -> str | None:
    explicit_awg = bool(AWG_MARKER_RE.search(question))
    approved_alias = _contains_alias(question, approved_aliases)
    kratno_scope_decision = _kratno_scope_decision(messages, question)
    if kratno_scope_decision is not None:
        return kratno_scope_decision
    if explicit_awg:
        return 'awg_marker'
    if personal_alias:
        return 'personal_alias'
    if CORPORATE_POSSESSIVE_RE.search(question):
        return 'awg_possessive_intent'
    if (
        CORPORATE_FIRST_PERSON_RE.search(question)
        and DELIVERY_QUESTION_RE.search(question)
        and not UNRELATED_COMPANY_RE.search(question)
    ):
        return 'awg_first_person_delivery_question'
    if (
        CORPORATE_FIRST_PERSON_RE.search(question)
        and GROUNDED_INTENT_RE.search(question)
        and not UNRELATED_COMPANY_RE.search(question)
    ):
        return 'awg_first_person_intent'
    if _has_prior_awg_anchor(messages):
        return 'confirmed_awg_continuation'
    if UNRELATED_COMPANY_RE.search(question):
        return None
    if approved_alias:
        return 'approved_alias'
    return None


def route_request(
    messages: list[dict],
    *,
    approved_aliases: tuple[str, ...] = (),
    personal_alias: bool = False,
) -> RouteDecision:
    """Choose an AWG GPT route without trusting client metadata."""
    question = latest_user_text(messages)
    if not question:
        return RouteDecision('out_of_scope', 'empty_request')

    memory = parse_memory_command(question)
    if memory is not None:
        return RouteDecision('memory_command', 'personal_memory', memory.operation)
    if POLICY_ATTACK_RE.search(question):
        return RouteDecision('out_of_scope', 'policy_extraction_or_override')
    if GREETING_RE.fullmatch(question):
        return RouteDecision('greeting_help', 'assistant_help')
    if (
        PROFILE_RE.search(question)
        and not GROUNDED_INTENT_RE.search(question)
        and not UNRELATED_COMPANY_RE.search(question)
    ):
        return RouteDecision('corporate_profile', 'approved_profile')
    if META_RE.search(question) and not GROUNDED_INTENT_RE.search(question):
        return RouteDecision('assistant_meta', 'assistant_identity')
    if needs_project_clarification(messages, approved_aliases, personal_alias):
        return RouteDecision('clarification', 'missing_awg_anchor')

    scope_decision = _grounded_scope_decision(messages, question, approved_aliases, personal_alias)
    if scope_decision is not None:
        return RouteDecision('confluence_grounded', scope_decision)
    return RouteDecision('out_of_scope', 'no_confirmed_awg_context')
