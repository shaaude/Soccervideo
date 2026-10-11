"""Alembic migrations apply cleanly and match the ORM models (PostgreSQL only)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

URL = os.environ.get("SOCCER_TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not URL.startswith("postgresql"), reason="set SOCCER_TEST_DATABASE_URL to a PostgreSQL DB")
REPO = Path(__file__).resolve().parents[1]


def alembic(*args):
    env = {**os.environ, "SOCCER_DATABASE_URL": URL}
    return subprocess.run([sys.executable, "-m", "alembic", *args], cwd=REPO, env=env, capture_output=True, text=True)


def test_upgrade_check_downgrade():
    from sqlalchemy import create_engine, text
    eng = create_engine(URL)
    with eng.begin() as c:
        c.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    eng.dispose()
    r = alembic("upgrade", "head")
    assert r.returncode == 0, r.stderr
    r = alembic("check")
    assert r.returncode == 0 and "No new upgrade operations" in (r.stdout + r.stderr), r.stdout + r.stderr
    assert alembic("downgrade", "base").returncode == 0
    assert alembic("upgrade", "head").returncode == 0
