from __future__ import annotations

import os


HERMES_ENABLED = os.getenv('AWG_HERMES_ENABLED', 'false').lower() == 'true'
HERMES_MODEL_IDS = frozenset(item.strip() for item in os.getenv('AWG_HERMES_MODEL_IDS', '').split(',') if item.strip())
HERMES_PROVISIONER_URL = os.getenv('AWG_HERMES_PROVISIONER_URL', 'http://awg-hermes-provisioner:8788').rstrip('/')
HERMES_BROKER_URL = os.getenv('AWG_HERMES_BROKER_URL', 'http://openwebui:8080').rstrip('/')
HERMES_CONTROL_SECRET = os.getenv('AWG_HERMES_CONTROL_SECRET', '')
HERMES_REQUEST_TIMEOUT = int(os.getenv('AWG_HERMES_REQUEST_TIMEOUT', '900'))
HERMES_QWEN_BASE_URL = os.getenv('AWG_HERMES_QWEN_BASE_URL', 'http://llama:8000/v1').rstrip('/')
HERMES_QWEN_API_KEY = os.getenv('AWG_HERMES_QWEN_API_KEY', '')
HERMES_QWEN_MODEL = os.getenv('AWG_HERMES_QWEN_MODEL', '')
HERMES_ARTIFACT_MAX_BYTES = int(os.getenv('AWG_HERMES_ARTIFACT_MAX_BYTES', str(50 * 1024 * 1024)))
HERMES_ALLOWED_ARTIFACT_EXTENSIONS = frozenset({'pdf', 'docx', 'xlsx', 'pptx'})
HERMES_ALLOWED_ATTACHMENT_EXTENSIONS = HERMES_ALLOWED_ARTIFACT_EXTENSIONS


def validate_hermes_settings() -> None:
    if HERMES_ENABLED and len(HERMES_CONTROL_SECRET) < 32:
        raise RuntimeError('AWG_HERMES_CONTROL_SECRET must contain at least 32 characters')
    if HERMES_ENABLED and not HERMES_MODEL_IDS:
        raise RuntimeError('AWG_HERMES_MODEL_IDS must name at least one existing OpenWebUI model')
    if HERMES_ENABLED and (not HERMES_QWEN_API_KEY or not HERMES_QWEN_MODEL):
        raise RuntimeError('Hermes Qwen proxy requires a server credential and fixed model')
