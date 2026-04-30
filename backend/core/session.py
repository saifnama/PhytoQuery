import base64
import hashlib
import hmac
import os
import secrets
from functools import lru_cache
from typing import Optional

from fastapi import Request, Response

SESSION_COOKIE_NAME = "pq_session"
SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 180
SESSION_SECRET_FILE = os.path.join(os.getcwd(), "data", ".session-signing-secret")


@lru_cache(maxsize=1)
def _get_session_secret() -> bytes:
    env_secret = os.getenv("PHYTOQUERY_SESSION_SECRET")
    if env_secret:
        return env_secret.encode("utf-8")

    os.makedirs(os.path.dirname(SESSION_SECRET_FILE), exist_ok=True)
    if os.path.isfile(SESSION_SECRET_FILE):
        with open(SESSION_SECRET_FILE, "rb") as secret_file:
            return secret_file.read().strip()

    secret = secrets.token_hex(32).encode("utf-8")
    with open(SESSION_SECRET_FILE, "wb") as secret_file:
        secret_file.write(secret)
    return secret


def _sign_session_id(session_id: str) -> str:
    digest = hmac.new(_get_session_secret(), session_id.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")


def _encode_session_cookie(session_id: str) -> str:
    return f"{session_id}.{_sign_session_id(session_id)}"


def _decode_session_cookie(cookie_value: Optional[str]) -> Optional[str]:
    if not cookie_value or "." not in cookie_value:
        return None

    session_id, signature = cookie_value.rsplit(".", 1)
    if not session_id or not signature:
        return None

    expected = _sign_session_id(session_id)
    if not hmac.compare_digest(expected, signature):
        return None
    return session_id


def get_session_id(request: Request) -> Optional[str]:
    return _decode_session_cookie(request.cookies.get(SESSION_COOKIE_NAME))


def attach_session_cookie(response: Response, request: Request, session_id: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=_encode_session_cookie(session_id),
        max_age=SESSION_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path="/",
    )


def get_or_set_session_id(request: Request, response: Response) -> str:
    session_id = get_session_id(request)
    if session_id:
        return session_id

    session_id = f"sess_{secrets.token_urlsafe(24)}"
    attach_session_cookie(response, request, session_id)
    return session_id
