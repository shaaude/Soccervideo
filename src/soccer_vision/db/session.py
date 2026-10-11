from __future__ import annotations

import os

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker


def database_url() -> str:
    url = os.environ.get("SOCCER_DATABASE_URL")
    if not url:
        raise RuntimeError("Set SOCCER_DATABASE_URL, e.g. postgresql+psycopg://soccer:<password>@127.0.0.1:5432/soccervision")
    return url


def make_engine(url: str | None = None):
    url = url or database_url()
    eng = create_engine(url, future=True, pool_pre_ping=True)
    if url.startswith("sqlite"):
        @event.listens_for(eng, "connect")
        def _fk(dbapi_conn, _):  # enforce foreign keys in SQLite test databases
            dbapi_conn.execute("PRAGMA foreign_keys=ON")
    return eng


def make_sessionmaker(engine):
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
