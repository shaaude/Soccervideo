"""API + schema tests. Runs on SQLite always, and on PostgreSQL when SOCCER_TEST_DATABASE_URL is set."""
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from soccer_vision.api.app import create_app
from soccer_vision.db.models import (AccessAudit, Base, EventReview, Player, PlayerAppearance, Team,
                                     TeamRatingSnapshot)
from soccer_vision.db.session import make_engine

KEYS = "adm:admin:alice,rev:reviewer:bob,coa:coach:carol,gua:guardian:gil,ana:analyst:ann"
H = {r: {"X-API-Key": k} for r, k in [("admin", "adm"), ("reviewer", "rev"), ("coach", "coa"), ("guardian", "gua"), ("analyst", "ana")]}


def _urls():
    yield "sqlite"
    if os.environ.get("SOCCER_TEST_DATABASE_URL"):
        yield "postgres"


@pytest.fixture(params=list(_urls()))
def env(request, tmp_path, synthetic_video, monkeypatch):
    url = f"sqlite:///{tmp_path / 't.db'}" if request.param == "sqlite" else os.environ["SOCCER_TEST_DATABASE_URL"]
    engine = make_engine(url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    repo = tmp_path / "repo"
    (repo / "data" / "normalized").mkdir(parents=True)
    (repo / "config").mkdir()
    shutil.copy(synthetic_video, repo / "data" / "normalized" / "m1.mp4")
    with Session(engine) as s, s.begin():
        s.add_all([Team(team_id="A", team_name="Alpha"), Team(team_id="B", team_name="Beta"),
                   Player(player_id="p1", display_name="P1"), Player(player_id="p2", display_name="P2")])
    monkeypatch.setenv("SOCCER_API_KEYS", KEYS)
    monkeypatch.setenv("SOCCER_MEDIA_SECRET", "test-secret")

    def fake_process(repo_, match_id, video, cfg, manifest=None, stop_after=None, force=False, on_stage=None):
        """Stands in for the GPU pipeline: writes a run directory exactly like the real runner."""
        on_stage("detect_track", "detecting")
        run = Path(repo_) / "data" / "processed" / match_id / "run123"
        run.mkdir(parents=True, exist_ok=True)
        (Path(repo_) / "config" / f"{match_id}.json").write_text(json.dumps({
            "match_id": match_id, "video_id": "m1-cam", "normalized_fps": 25, "camera_id": "cam"}))
        pd.DataFrame({"event_id": ["m1-transfer-000001", "m1-transfer-000002"], "match_id": "m1",
                      "event_type": "ball_transfer_candidate", "video_time_s": [2.0, 6.0],
                      "initiator_track_id": [1, 2], "receiver_track_id": [2, 1],
                      "rule_version": "contact_v2"}).to_csv(run / "event_candidates.csv", index=False)
        state = {"run_id": "run123", "match_manifest": f"config/{match_id}.json", "video": video,
                 "basis": {"video_sha256": "x", "code_revision": "test", "weights": {}},
                 "stages": {"detect_track": {"status": "SUCCEEDED", "records": 10, "elapsed_seconds": 1.0},
                            "temporal_model": {"status": "SKIPPED", "reason": "not enabled"}}}
        (run / "manifest.json").write_text(json.dumps(state))
        return state

    monkeypatch.setattr("soccer_vision.pipeline.runner.process", fake_process)
    app = create_app(engine, repo, executor=lambda fn, *a: fn(*a))
    yield TestClient(app), engine
    engine.dispose()


def test_full_review_flow(env):
    c, engine = env
    assert c.post("/api/v1/matches", json={"match_id": "m1"}).status_code == 401
    assert c.post("/api/v1/matches", json={"match_id": "m1"}, headers=H["coach"]).status_code == 403
    r = c.post("/api/v1/matches", headers=H["admin"], json={"match_id": "m1", "match_date": "2026-10-01",
                                                            "home_team_id": "A", "away_team_id": "B", "play_format": "9v9"})
    assert r.status_code == 201
    with Session(engine) as s, s.begin():
        s.add_all([PlayerAppearance(match_id="m1", team_id="A", player_id="p1", reviewed_minutes_observed=45,
                                    identity_reviewed=True),
                   PlayerAppearance(match_id="m1", team_id="B", player_id="p2")])
        s.add_all([TeamRatingSnapshot(team_id="B", rating_system="elo_v1", rating=1510,
                                      effective_at=datetime(2026, 9, 20, tzinfo=timezone.utc),
                                      input_cutoff=datetime(2026, 9, 19, tzinfo=timezone.utc)),
                   TeamRatingSnapshot(team_id="B", rating_system="elo_v1", rating=1600,  # post-match: must be ignored
                                      effective_at=datetime(2026, 10, 2, tzinfo=timezone.utc),
                                      input_cutoff=datetime(2026, 10, 2, tzinfo=timezone.utc))])
    bad = c.post("/api/v1/matches/m1/videos", headers=H["admin"],
                 json={"video_id": "v-out", "file_path": "../../etc/passwd", "video_use_permission": "granted"})
    assert bad.status_code == 422
    assert c.post("/api/v1/matches/m1/videos", headers=H["admin"],
                  json={"video_id": "m1-pending", "file_path": "data/normalized/m1.mp4",
                        "video_use_permission": "pending"}).status_code == 201
    assert c.post("/api/v1/matches/m1/process", headers=H["admin"], json={"video_id": "m1-pending"}).status_code == 422

    # The same file cannot be registered twice for a match (unique sha256): 409.
    r = c.post("/api/v1/matches/m1/videos", headers=H["admin"],
               json={"video_id": "m1-cam", "file_path": "data/normalized/m1.mp4", "video_use_permission": "granted"})
    assert r.status_code == 409
    from soccer_vision.db.models import MatchVideo
    with Session(engine) as s, s.begin():
        s.delete(s.get(MatchVideo, "m1-pending"))
    assert c.post("/api/v1/matches/m1/videos", headers=H["admin"],
                  json={"video_id": "m1-cam", "file_path": "data/normalized/m1.mp4",
                        "video_use_permission": "granted"}).status_code == 201
    job = c.post("/api/v1/matches/m1/process", headers=H["admin"], json={"video_id": "m1-cam"}).json()
    j = c.get(f"/api/v1/jobs/{job['job_id']}", headers=H["admin"]).json()
    assert j["state"] == "completed" and j["run_id"] == "run123"
    assert {p["stage"]: p["status"] for p in j["provenance"]}["temporal_model"] == "SKIPPED"

    assert len(c.get("/api/v1/matches/m1/events", headers=H["reviewer"]).json()) == 2
    assert c.get("/api/v1/matches/m1/events", headers=H["coach"]).json() == []   # no unreviewed hypotheses
    assert c.get("/api/v1/matches/m1/events", headers=H["guardian"]).status_code == 403

    eid = "m1-transfer-000001"
    r = c.patch(f"/api/v1/events/{eid}", headers=H["reviewer"],
                json={"expected_version": 0, "review_status": "verified"})
    assert r.status_code == 422 and "machine-only" in r.json()["detail"]
    r = c.patch(f"/api/v1/events/{eid}", headers=H["reviewer"],
                json={"expected_version": 0, "event_type": "pass", "outcome": "completed", "review_status": "verified"})
    assert r.status_code == 422 and "needs an actor" in r.json()["detail"]
    r = c.patch(f"/api/v1/events/{eid}", headers=H["reviewer"],
                json={"expected_version": 0, "event_type": "pass", "outcome": "completed", "review_status": "verified",
                      "actor_player_id": "p9"})
    assert r.status_code == 422
    r = c.patch(f"/api/v1/events/{eid}", headers=H["reviewer"],
                json={"expected_version": 0, "event_type": "pass", "outcome": "completed", "review_status": "verified",
                      "actor_player_id": "p1", "notes": "clear pass"})
    assert r.status_code == 200 and r.json()["version"] == 1
    assert c.patch(f"/api/v1/events/{eid}", headers=H["reviewer"],
                   json={"expected_version": 0, "outcome": "incomplete"}).status_code == 409
    r = c.post("/api/v1/matches/m1/events", headers=H["reviewer"],
               json={"video_id": "m1-cam", "event_type": "shot", "outcome": "goal", "video_time_seconds": 9.0,
                     "actor_player_id": "p1"})
    assert r.status_code == 201

    prof = c.get("/api/v1/players/p1/profile", headers=H["coach"]).json()
    assert prof["verified_only"] and prof["profile"]["completed_passes"] == 1 and prof["profile"]["goals"] == 1
    assert prof["profile"]["reviewed_minutes"] == 45.0 and prof["profile"]["shots_per90"] == 2.0
    assert prof["coverage"]["excluded_not_confirmed"] == 0

    clips = c.get("/api/v1/players/p1/clips", headers=H["coach"]).json()
    assert len(clips) == 2
    assert c.get(clips[0]["url"]).status_code == 200
    assert c.get(clips[0]["url"].replace("sig=", "sig=0")).status_code == 403

    ana = c.get("/api/v1/teams/A/analysis", headers=H["analyst"]).json()
    m = ana["matches"][0]
    assert m["opponent_pre_match_rating"]["rating"] == 1510   # post-match snapshot ignored
    assert m["verified_event_counts"] == {"pass": 1, "shot": 1}

    with Session(engine) as s:
        assert s.scalar(select(func.count()).select_from(EventReview)) == 2
        assert s.scalar(select(func.count()).select_from(AccessAudit)) > 10
