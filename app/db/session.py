"""SQLAlchemy engine/session setup.

The engine is created lazily (module-level singleton created on first use)
so importing this module — e.g. for FastAPI route type hints, or in a
context with no live Postgres instance — never triggers a connection.
"""
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

_engine = None
_SessionLocal = None

# Replace each pooled connection after 30 minutes. Without this the app's
# connections lived forever, and each server-side Postgres process kept the
# memory its past (spatial) queries had used: four idle connections had
# grown to ~1.65 GB each, ~7 GB of billed database RAM in total. A recycled
# connection hands that memory back. pool_pre_ping already makes any
# connection closed from the server side harmless.
POOL_RECYCLE_SECONDS = 1800


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            get_settings().database_url,
            pool_pre_ping=True,
            pool_recycle=POOL_RECYCLE_SECONDS,
            future=True,
        )
    return _engine


def get_sessionmaker():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, future=True)
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope for a series of operations."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()
