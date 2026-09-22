"""Login/signup for the dashboard login screen.

Two layers, same split as the rest of the suite: `app/auth/security.py`
is tested directly with no DB and no app (hashing, token round-trip).
The routes are tested through TestClient against a real SQLAlchemy
session — an in-memory SQLite engine holding just the `users` table
(it has no Geometry column, so it doesn't need PostGIS), wired in via
a `get_db` dependency override. This keeps the whole suite DB-free and
portable — no test here depends on a live Postgres instance.
"""
import time

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth.security import create_access_token, decode_access_token, hash_password, verify_password
from app.db.models import Base, User
from app.db.session import get_db
from app.main import app


# --- security.py: hashing -------------------------------------------------

def test_hash_password_round_trips_through_verify():
    h = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", h)


def test_verify_password_rejects_the_wrong_password():
    h = hash_password("correct horse battery staple")
    assert not verify_password("wrong password", h)


def test_hash_password_salts_so_two_hashes_of_the_same_password_differ():
    """A fixed salt would mean identical passwords produce identical
    hashes, letting an attacker with DB read access spot reused
    passwords across accounts. Real salting prevents that."""
    a = hash_password("same-password")
    b = hash_password("same-password")
    assert a != b
    assert verify_password("same-password", a)
    assert verify_password("same-password", b)


def test_verify_password_never_raises_on_garbage_stored_hash():
    """A corrupted or foreign-format stored_hash must fail closed, not
    crash the login endpoint with a 500."""
    assert verify_password("anything", "not-a-real-hash") is False
    assert verify_password("anything", "") is False


# --- security.py: tokens ---------------------------------------------------

def test_create_access_token_round_trips_through_decode():
    token = create_access_token(user_id=42, username="ops1")
    payload = decode_access_token(token)
    assert payload["sub"] == "42"
    assert payload["username"] == "ops1"


def test_decode_access_token_rejects_a_tampered_token():
    token = create_access_token(user_id=1, username="ops1")
    tampered = token[:-4] + ("0" if token[-1] != "0" else "1") + token[-3:]
    with pytest.raises(jwt.PyJWTError):
        decode_access_token(tampered)


def test_decode_access_token_rejects_an_expired_token(monkeypatch):
    import app.auth.security as sec

    monkeypatch.setattr(sec.get_settings(), "auth_token_ttl_hours", 0)
    token = create_access_token(user_id=1, username="ops1")
    time.sleep(1.1)
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(token)


# --- routes: signup / login / me ------------------------------------------

@pytest.fixture()
def client():
    # StaticPool + check_same_thread=False: a bare sqlite:///:memory:
    # gives each new connection its own empty database, so the table
    # created here would vanish before the first request's connection
    # (TestClient runs requests through Starlette's own thread) could
    # see it — same failure shape as a real dropped DB, caught by
    # app.main's OperationalError handler and reported as a 503.
    engine = create_engine(
        "sqlite:///:memory:", future=True,
        connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine, tables=[User.__table__])
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    def _override_get_db():
        session = TestingSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override_get_db
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.pop(get_db, None)


def _admin_payload(username, **overrides):
    payload = {
        "role": "admin", "full_name": "Test Administrator", "username": username,
        "password": "secret123", "government_id": "ADMIN-TEST-0001",
    }
    payload.update(overrides)
    return payload


def _state_payload(username, state="Odisha", **overrides):
    payload = {
        "role": "state", "full_name": "Test State Officer", "username": username,
        "password": "secret123", "government_id": "STATE-TEST-0001", "state": state,
    }
    payload.update(overrides)
    return payload


def test_signup_creates_a_real_admin_account_and_returns_a_usable_token(client):
    r = client.post("/auth/signup", json=_admin_payload("newops"))
    assert r.status_code == 201
    body = r.json()
    assert body["username"] == "newops"
    assert body["role"] == "admin"
    assert body["state"] is None
    assert body["token_type"] == "bearer"

    me = client.get("/auth/me", headers={"Authorization": "Bearer " + body["access_token"]})
    assert me.status_code == 200
    assert me.json()["username"] == "newops"
    assert me.json()["role"] == "admin"


def test_signup_creates_a_real_state_account_scoped_to_its_state(client):
    r = client.post("/auth/signup", json=_state_payload("newstateops", state="Rajasthan"))
    assert r.status_code == 201
    body = r.json()
    assert body["role"] == "state"
    assert body["state"] == "Rajasthan"


def test_signup_rejects_a_state_account_with_no_state_selected(client):
    r = client.post("/auth/signup", json=_state_payload("nostateops", state=None))
    assert r.status_code == 422


def test_signup_rejects_an_admin_account_that_also_sets_a_state(client):
    r = client.post("/auth/signup", json=_admin_payload("adminwithstate", state="Odisha"))
    assert r.status_code == 422


def test_signup_rejects_a_state_that_is_not_a_real_indian_state(client):
    r = client.post("/auth/signup", json=_state_payload("faketstate", state="Narnia"))
    assert r.status_code == 422


def test_signup_rejects_a_duplicate_username_case_insensitively(client):
    client.post("/auth/signup", json=_admin_payload("dupeops"))
    r = client.post("/auth/signup", json=_admin_payload("DupeOps"))
    assert r.status_code == 409


def test_signup_rejects_a_too_short_password(client):
    r = client.post("/auth/signup", json=_admin_payload("shortpw", password="abc"))
    assert r.status_code == 422


def test_signup_rejects_an_invalid_username(client):
    r = client.post("/auth/signup", json=_admin_payload("a b!"))
    assert r.status_code == 422


def test_signup_rejects_a_missing_government_id(client):
    r = client.post("/auth/signup", json=_admin_payload("nogovid", government_id=""))
    assert r.status_code == 422


def test_login_succeeds_with_the_right_password_and_matching_role(client):
    client.post("/auth/signup", json=_admin_payload("loginok"))
    r = client.post("/auth/login", json={"role": "admin", "username": "loginok", "password": "secret123"})
    assert r.status_code == 200
    assert "access_token" in r.json()


def test_login_fails_with_the_wrong_password(client):
    client.post("/auth/signup", json=_admin_payload("loginbad"))
    r = client.post("/auth/login", json={"role": "admin", "username": "loginbad", "password": "nope-nope"})
    assert r.status_code == 401


def test_login_fails_for_a_username_that_does_not_exist(client):
    r = client.post("/auth/login", json={"role": "admin", "username": "ghost-user", "password": "whatever1"})
    assert r.status_code == 401


def test_login_error_message_does_not_reveal_which_field_was_wrong(client):
    """Same message for 'no such user' and 'wrong password' — otherwise
    the endpoint becomes a username-enumeration oracle."""
    client.post("/auth/signup", json=_admin_payload("revealtest"))
    wrong_pw = client.post("/auth/login", json={"role": "admin", "username": "revealtest", "password": "nope-nope"})
    no_user = client.post("/auth/login", json={"role": "admin", "username": "no-such-user", "password": "nope-nope"})
    assert wrong_pw.json()["detail"] == no_user.json()["detail"]


def test_state_account_cannot_log_in_through_the_administrator_door(client):
    client.post("/auth/signup", json=_state_payload("odishaops", state="Odisha"))
    r = client.post("/auth/login", json={"role": "admin", "username": "odishaops", "password": "secret123"})
    assert r.status_code == 403
    assert "State" in r.json()["detail"]


def test_administrator_account_cannot_log_in_through_the_state_door(client):
    client.post("/auth/signup", json=_admin_payload("adminops2"))
    r = client.post("/auth/login", json={"role": "state", "username": "adminops2", "password": "secret123"})
    assert r.status_code == 403
    assert "National Admin" in r.json()["detail"]


def test_state_account_logs_in_correctly_through_the_state_door(client):
    client.post("/auth/signup", json=_state_payload("odishaops2", state="Odisha"))
    r = client.post("/auth/login", json={"role": "state", "username": "odishaops2", "password": "secret123"})
    assert r.status_code == 200
    assert r.json()["state"] == "Odisha"


def test_me_requires_a_token(client):
    r = client.get("/auth/me")
    assert r.status_code == 401


def test_me_rejects_a_garbage_token(client):
    r = client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 401


def test_password_is_never_stored_or_returned_in_plaintext(client):
    r = client.post("/auth/signup", json=_admin_payload("plaintextcheck", password="hunter2xyz"))
    assert "hunter2xyz" not in r.text
