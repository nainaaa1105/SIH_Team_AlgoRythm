"""Password hashing and session tokens for the login/signup endpoints.

No new dependency: PBKDF2-HMAC-SHA256 (stdlib `hashlib`) instead of
bcrypt/argon2, and PyJWT (already installed for other parts of the
stack) for the session token. PBKDF2 at this iteration count is the
same primitive Django's default hasher uses — not a toy scheme.
"""
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import jwt

from app.config import get_settings

_ALGO = "pbkdf2_sha256"
_ITERATIONS = 260_000
_SALT_BYTES = 16


def hash_password(password: str) -> str:
    salt = secrets.token_hex(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), _ITERATIONS)
    return f"{_ALGO}${_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algo, iterations, salt, hex_digest = stored_hash.split("$")
        if algo != _ALGO:
            return False
        computed = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iterations)
        )
        return hmac.compare_digest(computed.hex(), hex_digest)
    except (ValueError, AttributeError):
        return False


def create_access_token(user_id: int, username: str) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": now,
        "exp": now + timedelta(hours=settings.auth_token_ttl_hours),
    }
    return jwt.encode(payload, settings.auth_secret_key, algorithm="HS256")


def decode_access_token(token: str) -> Dict[str, Any]:
    """Raises jwt.PyJWTError (ExpiredSignatureError, InvalidTokenError,
    ...) on anything wrong — callers turn that into a 401, not a 500."""
    settings = get_settings()
    return jwt.decode(token, settings.auth_secret_key, algorithms=["HS256"])
