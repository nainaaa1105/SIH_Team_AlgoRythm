"""Real login/signup for the dashboard login screen (static/index.html
#login-screen). Before this, "Sign In" dismissed the overlay on any
click regardless of what was typed — no account was created or checked
against anything. These endpoints back that screen with a real `users`
table (see app/db/models.py::User), PBKDF2-hashed passwords, and a
signed session token the frontend stores and sends back as a Bearer
token.

Two account roles, picked on both the signup and login screens:
- "admin": sees the all-India dashboard, unscoped.
- "state": represents one Indian state; the dashboard filters itself
  to that state on login (see static/index.html::enterDashboard).

A login's requested role must match the account's actual role — an
admin account cannot sign in through the "State" door and vice versa,
per the access-control model this was built for.
"""
import re
from typing import Literal, Optional

import jwt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth.security import create_access_token, decode_access_token, hash_password, verify_password
from app.db.models import User
from app.db.session import get_db

router = APIRouter(prefix="/auth", tags=["auth"])

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")
_bearer = HTTPBearer(auto_error=False)


def _real_state_names() -> Optional[list]:
    """The same state list the dashboard's own filter dropdown is built
    from (geospatial.admin_boundaries, backed by the real district
    boundary file) — never a hand-typed list a signup could drift out
    of sync with. Returns None if the boundary resolver isn't available
    at all, in which case signup skips state validation rather than
    blocking every state signup on an unrelated data-loading problem.
    """
    try:
        from geospatial.admin_boundaries import available, state_names

        if not available():
            return None
        return state_names()
    except Exception:  # noqa: BLE001
        return None


class SignupRequest(BaseModel):
    role: Literal["admin", "state"]
    full_name: str = Field(min_length=2, max_length=120)
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=6, max_length=128)
    government_id: str = Field(min_length=3, max_length=64)
    state: Optional[str] = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def _state_matches_role(self):
        if self.role == "state" and not (self.state and self.state.strip()):
            raise ValueError("A state account must select which state it represents.")
        if self.role == "admin" and self.state:
            raise ValueError("An administrator account does not represent a single state.")
        return self


class LoginRequest(BaseModel):
    role: Literal["admin", "state"]
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    full_name: str
    role: str
    state: Optional[str] = None


class UserOut(BaseModel):
    id: int
    username: str
    full_name: str
    role: str
    state: Optional[str] = None


@router.post("/signup", response_model=TokenResponse, status_code=201)
def signup(body: SignupRequest, db: Session = Depends(get_db)):
    if not _USERNAME_RE.match(body.username):
        raise HTTPException(
            status_code=422,
            detail="Username must be 3-32 characters: letters, numbers, underscore, dot or hyphen only.",
        )

    state = body.state.strip() if body.state else None
    if body.role == "state":
        real_states = _real_state_names()
        if real_states is not None and state not in real_states:
            raise HTTPException(
                status_code=422,
                detail=f"'{state}' is not a recognised Indian state/UT on this system.",
            )

    # Case-insensitive uniqueness — "Ops" and "ops" are the same account
    # to a human operator even though the column comparison isn't.
    existing = db.query(User).filter(func.lower(User.username) == body.username.lower()).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail="That username is already taken.")

    user = User(
        username=body.username,
        password_hash=hash_password(body.password),
        full_name=body.full_name.strip(),
        government_id=body.government_id.strip(),
        role=body.role,
        state=state,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id, user.username)
    return TokenResponse(
        access_token=token, username=user.username, full_name=user.full_name,
        role=user.role, state=user.state,
    )


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(func.lower(User.username) == body.username.lower()).first()
    # Same generic message whether the username doesn't exist or the
    # password is wrong — telling them apart lets an attacker enumerate
    # valid usernames.
    invalid = HTTPException(status_code=401, detail="Incorrect username or password.")
    if user is None or not verify_password(body.password, user.password_hash):
        raise invalid

    if user.role != body.role:
        # A real, distinct business rule rather than a credential secret
        # — telling an administrator they picked the wrong door is not
        # an enumeration risk the way "wrong password" is, and hiding it
        # would just leave them stuck with no way to know why.
        wanted = "a National Admin" if body.role == "admin" else "a State Admin"
        actual = "a National Admin" if user.role == "admin" else "a State Admin"
        raise HTTPException(
            status_code=403,
            detail=f"This account is registered as {actual} account, not {wanted} account. "
                   f"Use the {('National Admin' if user.role == 'admin' else 'State Admin')} login instead.",
        )

    token = create_access_token(user.id, user.username)
    return TokenResponse(
        access_token=token, username=user.username, full_name=user.full_name,
        role=user.role, state=user.state,
    )


def current_user(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI dependency other routers can import to require a logged-in
    operator — not wired onto the existing dashboard/geospatial routes
    yet (that would be a much larger, separately-scoped change), but
    available for /auth/me and any route that opts in."""
    if creds is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    try:
        payload = decode_access_token(creds.credentials)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired, please log in again.")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid session token.")

    user = db.get(User, int(payload["sub"]))
    if user is None:
        raise HTTPException(status_code=401, detail="Account no longer exists.")
    return user


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    """Lets the frontend check a token it already has in localStorage is
    still valid on page load, instead of re-showing the login screen
    every reload."""
    return UserOut(id=user.id, username=user.username, full_name=user.full_name, role=user.role, state=user.state)
