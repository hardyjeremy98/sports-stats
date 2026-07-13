from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from pitchlab_server.settings import get_settings


class Base(DeclarativeBase):
    pass


_engine = None
_session_factory = None


def get_engine():
    global _engine, _session_factory
    if _engine is None:
        url = get_settings().database_url
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args)
        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def init_db() -> None:
    import pitchlab_server.models  # noqa: F401

    engine = get_engine()
    Base.metadata.create_all(engine)
    _micro_migrations(engine)


def _micro_migrations(engine) -> None:
    """create_all only creates missing tables; columns added to existing models
    are patched in here (no migration framework — sqlite/postgres additive only)."""
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    if "videos" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("videos")}
        if "gt_path" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE videos ADD COLUMN gt_path TEXT"))


def session() -> Session:
    get_engine()
    return _session_factory()


def get_db():
    """FastAPI dependency."""
    db = session()
    try:
        yield db
    finally:
        db.close()
