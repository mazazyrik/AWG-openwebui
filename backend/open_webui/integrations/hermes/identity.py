from __future__ import annotations

import base64
import hashlib
import hmac
import time
from dataclasses import dataclass
from uuid import uuid4

from fastapi import HTTPException, status

from open_webui.integrations.hermes.settings import HERMES_CONTROL_SECRET, HERMES_REQUEST_TIMEOUT
from open_webui.models.files import Files
from open_webui.utils.access_control.files import has_access_to_file
from open_webui.utils.json_codec import JSONCodec

TOKEN_TTL_SECONDS = HERMES_REQUEST_TIMEOUT + 300
MAX_CLOCK_SKEW_SECONDS = 60


@dataclass(frozen=True)
class HermesPrincipal:
    user_id: str
    scope_id: str
    jti: str
    chat_id: str
    message_id: str
    run_id: str
    allowed_file_ids: frozenset[str]
    expires_at: int


def user_scope_id(user_id: str) -> str:
    return hashlib.sha256(f'awg-hermes-user:{user_id}'.encode()).hexdigest()[:32]


def _sign(value: bytes) -> str:
    return hmac.new(HERMES_CONTROL_SECRET.encode(), value, hashlib.sha256).hexdigest()


def issue_principal_token(
    user_id: str,
    *,
    chat_id: str,
    message_id: str,
    run_id: str,
    allowed_file_ids: list[str],
    now: int | None = None,
) -> tuple[str, str]:
    issued_at = now or int(time.time())
    jti = str(uuid4())
    payload = {
        'sub': user_id,
        'scope': user_scope_id(user_id),
        'iat': issued_at,
        'exp': issued_at + TOKEN_TTL_SECONDS,
        'aud': 'awg-hermes-broker',
        'jti': jti,
        'chat': chat_id,
        'message': message_id,
        'run': run_id,
        'files': sorted(set(allowed_file_ids)),
    }
    encoded = base64.urlsafe_b64encode(JSONCodec.dumps(payload).encode()).rstrip(b'=')
    return f'{encoded.decode()}.{_sign(encoded)}', jti


def verify_principal_token(token: str, *, now: int | None = None) -> HermesPrincipal:
    try:
        encoded, signature = token.split('.', 1)
        encoded_bytes = encoded.encode()
        if not hmac.compare_digest(signature, _sign(encoded_bytes)):
            raise ValueError('signature')
        padding = '=' * (-len(encoded) % 4)
        payload = JSONCodec.loads(base64.urlsafe_b64decode(encoded + padding).decode())
        current = now or int(time.time())
        if payload.get('aud') != 'awg-hermes-broker':
            raise ValueError('audience')
        if int(payload.get('iat', 0)) > current + MAX_CLOCK_SKEW_SECONDS:
            raise ValueError('issued_at')
        if int(payload.get('exp', 0)) < current:
            raise ValueError('expired')
        user_id = str(payload['sub'])
        scope_id = str(payload['scope'])
        jti = str(payload['jti'])
        if not hmac.compare_digest(scope_id, user_scope_id(user_id)):
            raise ValueError('scope')
        file_ids = payload.get('files')
        if not isinstance(file_ids, list) or any(not isinstance(file_id, str) for file_id in file_ids):
            raise ValueError('files')
        return HermesPrincipal(
            user_id=user_id,
            scope_id=scope_id,
            jti=jti,
            chat_id=str(payload['chat']),
            message_id=str(payload['message']),
            run_id=str(payload['run']),
            allowed_file_ids=frozenset(file_ids),
            expires_at=int(payload['exp']),
        )
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid Hermes principal token')


def sign_control_request(method: str, path: str, body: bytes, timestamp: str) -> str:
    digest = hashlib.sha256(body).hexdigest()
    canonical = f'{timestamp}\n{method.upper()}\n{path}\n{digest}'.encode()
    return _sign(canonical)


def verify_control_request(method: str, path: str, body: bytes, timestamp: str, signature: str) -> None:
    try:
        request_time = int(timestamp)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid control timestamp')
    if abs(int(time.time()) - request_time) > MAX_CLOCK_SKEW_SECONDS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Expired control request')
    expected = sign_control_request(method, path, body, timestamp)
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid control signature')


async def get_authorized_file(file_id: str, user):
    file = await Files.get_file_by_id(file_id)
    if not file or (file.user_id != user.id and not await has_access_to_file(file_id, 'read', user)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='File not found')
    return file
