from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from open_webui.utils.json_codec import JSONCodec

log = logging.getLogger(__name__)


class ConfluenceSearchArguments(BaseModel):
    model_config = ConfigDict(extra='forbid')

    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=6, ge=1, le=8)


class AttachmentArguments(BaseModel):
    model_config = ConfigDict(extra='forbid')

    file_id: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_-]+$')


class PluginStageArguments(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str = Field(pattern=r'^[a-z][a-z0-9_-]{1,63}$')
    files: dict[str, str] = Field(min_length=2, max_length=64)


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    'search_confluence': {
        'type': 'object',
        'properties': {
            'query': {'type': 'string', 'minLength': 1, 'maxLength': 2000},
            'limit': {'type': 'integer', 'minimum': 1, 'maximum': 8, 'default': 6},
        },
        'required': ['query'],
        'additionalProperties': False,
    },
    'read_attachment': {
        'type': 'object',
        'properties': {
            'file_id': {'type': 'string', 'minLength': 1, 'maxLength': 128, 'pattern': '^[A-Za-z0-9_-]+$'},
        },
        'required': ['file_id'],
        'additionalProperties': False,
    },
    'stage_plugin': {
        'type': 'object',
        'properties': {
            'name': {'type': 'string', 'pattern': '^[a-z][a-z0-9_-]{1,63}$'},
            'files': {
                'type': 'object',
                'minProperties': 2,
                'maxProperties': 64,
                'additionalProperties': {'type': 'string', 'maxLength': 262144},
            },
        },
        'required': ['name', 'files'],
        'additionalProperties': False,
    },
}

_ARGUMENT_MODELS = {
    'search_confluence': ConfluenceSearchArguments,
    'read_attachment': AttachmentArguments,
    'stage_plugin': PluginStageArguments,
}


@dataclass(frozen=True)
class ToolDecision:
    allowed: bool
    reason: str
    arguments: BaseModel | None
    argument_hash: str


def authorize_tool(actor_scope: str, tool: str, arguments: dict[str, Any]) -> ToolDecision:
    argument_hash = hashlib.sha256(JSONCodec.dumps(arguments, sort_keys=True).encode()).hexdigest()
    model = _ARGUMENT_MODELS.get(tool)
    if model is None:
        decision = ToolDecision(False, 'unknown_tool', None, argument_hash)
    else:
        try:
            decision = ToolDecision(True, 'allowlisted', model.model_validate(arguments), argument_hash)
        except ValidationError:
            decision = ToolDecision(False, 'invalid_arguments', None, argument_hash)
    log.info(
        'hermes_tool_decision actor_scope=%s tool=%s argument_hash=%s allowed=%s reason=%s',
        actor_scope,
        tool,
        argument_hash,
        decision.allowed,
        decision.reason,
    )
    return decision
