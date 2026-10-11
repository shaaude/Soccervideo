"""Orchestrated pipeline (Step 24) end to end with a scripted detector: statuses, run hash, resume, gates."""
import json
import shutil

import pandas as pd
import pytest
import yaml

from soccer_vision.pipeline.runner import process
from soccer_vision.review.store import ReviewStore
from conftest import REPO, FakePredictor

NAMES = {0: "Player", 1: "GoalKeeper", 2: "Ball", 3: "Main Referee", 4: "Side Referee", 5: "Staff Member"}


def script(n=300):
    out = []
    for f in range(n):
        boxes = [[100, 100, 120, 180], [220, 100, 240, 180], [300, 20, 310, 60]]
        conf, cls = [0.9, 0.9, 0.8], [0, 0, 3]
        if f < 50 or 130 <= f < 170:
            owner = 110          # at player A's feet
        elif 60 <= f < 120 or f >= 180:
            owner = 230          # at player B's feet
        else:
            owner = None         # in flight
        if owner is not None:
            boxes.append([owner - 3, 177, owner + 3, 183]); conf.append(0.7); cls.append(2)
        else:
            boxes.append([170, 120, 176, 126]); conf.append(0.5); cls.append(2)
        out.append((boxes, conf, cls))
    return out


@pytest.fixture
def repo(tmp_path, synthetic_video, monkeypatch):
    r = tmp_path / "repo"
    for d in ("config/classmaps", "data/normalized", "models"):
        (r / d).mkdir(parents=True)
    shutil.copy(REPO / "config/event_taxonomy.yaml", r / "config")
    shutil.copy(REPO / "config/classmaps/soccernet_yolov8_smoke.yaml", r / "config/classmaps")
    shutil.copy(synthetic_video, r / "data/normalized/t1.mp4")
    (r / "models/fake.pt").write_bytes(b"weights-v1")
    cfg = yaml.safe_load((REPO / "config/pipeline.smoke.yaml").read_text())
    cfg["detection"]["player_weights"] = "models/fake.pt"
    cfg["detection"]["annotated_video"] = False
    (r / "config/pipeline.yaml").write_text(yaml.safe_dump(cfg))
    m = json.loads((REPO / "config/match001.json").read_text())
    m.update({"match_id": "t1", "video_id": "t1-cam", "normalized_video": "data/normalized/t1.mp4", "play_format": "9v9"})
    (r / "config/t1.json").write_text(json.dumps(m))
    calls = {"n": 0}

    def factory(weights, device, imgsz, conf):
        calls["n"] += 1
        return FakePredictor(NAMES, script())
    monkeypatch.setattr("soccer_vision.detection.export.UltralyticsPredictor", factory)
    return r, calls


def statuses(state):
    return {k: v["status"] for k, v in state["stages"].items()}


def test_pipeline_end_to_end_resume_and_review_gate(repo):
    r, calls = repo
    st = process(r, "t1", "data/normalized/t1.mp4", r / "config/pipeline.yaml")
    s = statuses(st)
    assert s == {"validate": "SUCCEEDED", "detect_track": "SUCCEEDED", "calibrate": "SKIPPED", "identify": "SUCCEEDED",
                 "candidates": "SUCCEEDED", "temporal_model": "SKIPPED", "highlights": "SUCCEEDED",
                 "review_queue": "SUCCEEDED", "aggregate": "SKIPPED"}
    assert "explicitly MISSING" in st["stages"]["calibrate"]["reason"]
    run_dir = r / "data/processed/t1" / st["run_id"]
    ev = pd.read_csv(run_dir / "event_candidates.csv")
    assert list(ev.initiator_track_id) == [1, 2, 1] and (ev.review_status == "unreviewed").all()
    assert len(pd.read_csv(run_dir / "highlights/clip_index.csv")) == 3
    assert (run_dir / "tracklet_team_clusters.csv").exists() and (run_dir / "identity_review_queue.csv").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["stages"]["detect_track"]["outputs"]["tracks"]["sha256"]

    # Rerun with identical inputs: nothing recomputed (detector not constructed again).
    n = calls["n"]
    st2 = process(r, "t1", "data/normalized/t1.mp4", r / "config/pipeline.yaml")
    assert st2["run_id"] == st["run_id"] and calls["n"] == n
    assert all("reused_at" in v for k, v in st2["stages"].items() if v["status"] == "SUCCEEDED")

    # A human confirms one event -> only aggregation reruns and produces verified-only profiles.
    store = ReviewStore(r / "data/annotations/t1", "t1")
    store.save_review("t1-transfer-000001", {"event_type": "pass", "outcome": "completed", "review_status": "verified",
                                             "actor_player_id": "p7"}, "rev", 0)
    st3 = process(r, "t1", "data/normalized/t1.mp4", r / "config/pipeline.yaml")
    assert statuses(st3)["aggregate"] == "SUCCEEDED" and calls["n"] == n
    prof = pd.read_csv(run_dir / "player_profiles.csv", dtype={"player_id": str})
    assert prof.player_id.tolist() == ["p7"] and prof.completed_passes.tolist() == [1]


def test_changed_weights_create_new_run(repo):
    r, _ = repo
    a = process(r, "t1", "data/normalized/t1.mp4", r / "config/pipeline.yaml", stop_after="validate")
    (r / "models/fake.pt").write_bytes(b"weights-v2")
    b = process(r, "t1", "data/normalized/t1.mp4", r / "config/pipeline.yaml", stop_after="validate")
    assert a["run_id"] != b["run_id"]


def test_missing_checkpoint_fails_fast(repo):
    r, _ = repo
    (r / "models/fake.pt").unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        process(r, "t1", "data/normalized/t1.mp4", r / "config/pipeline.yaml")


def test_fps_mismatch_fails_validation_and_is_recorded(repo):
    r, _ = repo
    m = json.loads((r / "config/t1.json").read_text())
    m["normalized_fps"] = 30
    (r / "config/t1.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="FPS"):
        process(r, "t1", "data/normalized/t1.mp4", r / "config/pipeline.yaml")
    man = next((r / "data/processed/t1").glob("*/manifest.json"))
    rec = json.loads(man.read_text())["stages"]["validate"]
    assert rec["status"] == "FAILED" and "FPS" in rec["failure_reason"]


def test_denied_permission_blocks_processing(repo):
    r, _ = repo
    m = json.loads((r / "config/t1.json").read_text())
    m["video_use_permission"] = "denied"
    (r / "config/t1.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="denied"):
        process(r, "t1", "data/normalized/t1.mp4", r / "config/pipeline.yaml")
