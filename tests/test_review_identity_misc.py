import numpy as np
import pandas as pd
import pytest

from soccer_vision.evaluation.metrics import event_metrics, identity_metrics
from soccer_vision.identity.assignments import validate_assignments
from soccer_vision.identity.team import kmeans
from soccer_vision.manifest import load_manifest, validate_manifest
from soccer_vision.review.store import ConflictError, ReviewStore, ReviewValidationError
from conftest import REPO


def candidates(n=2):
    return pd.DataFrame({"event_id": [f"m1-transfer-{i:06d}" for i in range(1, n + 1)], "match_id": "m1",
                         "event_type": "ball_transfer_candidate", "video_time_s": [1.0 + i for i in range(n)],
                         "initiator_track_id": 1, "receiver_track_id": 2, "rule_version": "contact_v2"})


def test_review_store_end_to_end(tmp_path):
    s = ReviewStore(tmp_path, "m1")
    assert s.sync_candidates(candidates()) == {"added": 2, "total": 2, "orphaned": 0}
    eid = "m1-transfer-000001"
    with pytest.raises(ReviewValidationError, match="machine-only"):
        s.save_review(eid, {"review_status": "verified"}, "rev1", 0)
    with pytest.raises(ReviewValidationError, match="needs actor"):
        s.save_review(eid, {"event_type": "pass", "outcome": "completed", "review_status": "verified"}, "rev1", 0)
    s.save_review(eid, {"event_type": "pass", "outcome": "completed", "review_status": "verified",
                        "actor_player_id": "12"}, "rev1", 0)
    with pytest.raises(ConflictError):  # stale version
        s.save_review(eid, {"review_notes": "x"}, "rev2", 0)
    # time correction on a model event becomes 'corrected'
    s.save_review(eid, {"video_time_s": 1.4, "review_status": "verified"}, "rev2", 1)
    row = ReviewStore(tmp_path, "m1").load().set_index("event_id").loc[eid]   # survives a restart
    assert row.review_status == "corrected" and row.actor_player_id == "12"
    assert float(row.predicted_video_time_s) == 1.0 and float(row.video_time_s) == 1.4 and int(row.version) == 2
    added = s.add_missed_event({"event_type": "shot", "outcome": "saved", "video_time_s": 5.0,
                                "actor_player_id": "UNKNOWN"}, "rev1")
    hist = s.audit_history()
    assert [h["action"] for h in hist] == ["review", "review", "add_missed_event"]
    assert hist[0]["previous"]["event_type"] == "ball_transfer_candidate"
    # re-sync with a candidate set that lost one event: review kept, flagged orphaned
    res = s.sync_candidates(candidates(2).iloc[1:])
    assert res["orphaned"] == 1 and added in set(s.load().event_id)


def test_reviewer_id_required(tmp_path):
    s = ReviewStore(tmp_path, "m1")
    s.sync_candidates(candidates(1))
    with pytest.raises(ReviewValidationError):
        s.save_review("m1-transfer-000001", {"review_status": "rejected"}, " ", 0)


def test_identity_constraints():
    tracks = pd.DataFrame({"track_id": [1, 1, 2, 2, 3], "frame_index": [0, 50, 40, 90, 200]})
    roster = pd.DataFrame({"player_id": ["p1", "p2"], "team_id": ["A", "A"], "jersey_number": ["7", "9"],
                           "start_date": ["2026-08-01", "2026-08-01"], "end_date": [None, "2026-09-01"]})
    a = pd.DataFrame({"video_id": "v", "run_id": "r", "segment_id": 0, "track_id": [1, 2, 3],
                      "player_id": ["p1", "p1", "p2"], "team_id": ["A", "A", "A"], "jersey_evidence": ["7", "", "9"],
                      "identity_confidence": [0.9, 0.9, 0.9], "review_status": "verified"})
    errs = validate_assignments(a, tracks, roster, "2026-10-01")
    assert any("overlap" in e for e in errs)            # p1 on tracks 1 and 2 simultaneously
    assert any("p2 not on an eligible roster" in e for e in errs)  # p2's membership ended
    a.loc[0, "jersey_evidence"] = "8"
    assert any("contradicts" in e for e in validate_assignments(a, tracks, roster, "2026-10-01"))


def test_kmeans_separates_two_kits():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 0.1, (10, 3)), rng.normal(5, 0.1, (10, 3))])
    lab, _, _ = kmeans(X, 2)
    assert len(set(lab[:10])) == 1 and len(set(lab[10:])) == 1 and lab[0] != lab[10]


def test_manifest_template_is_flagged_not_production_ready():
    m = load_manifest(REPO / "config/match001.json")
    assert validate_manifest(m).ok
    rep = validate_manifest(m, production=True)
    assert not rep.ok and any("unresolved" in e for e in rep.errors) and any("pending" in e for e in rep.errors)
    m["video_use_permission"] = "denied"
    assert not validate_manifest(m).ok


def test_event_metrics_counts_misses_and_false_positives():
    truth = pd.DataFrame({"match_id": "m", "event_type": ["pass", "pass", "shot"], "video_time_s": [1.0, 5.0, 9.0],
                          "actor_player_id": ["a", "b", "c"]})
    pred = pd.DataFrame({"match_id": "m", "event_type": ["pass", "pass", "pass"], "video_time_s": [1.3, 7.5, 5.2],
                         "actor_player_id": ["a", "x", "z"]})
    m = event_metrics(pred, truth, 1.0).set_index("event_type")
    assert (m.loc["pass", "true_positives"], m.loc["pass", "false_positives"], m.loc["pass", "false_negatives"]) == (2, 1, 0)
    assert m.loc["shot", "recall"] == 0.0
    assert m.loc["__all__", "joint_event_actor_f1"] == pytest.approx(2 * (1 / 3) * (1 / 3) / (2 / 3), abs=1e-3)


def test_identity_metrics_wrong_high_confidence():
    truth = pd.DataFrame({"video_id": "v", "track_id": [1, 2, 3], "player_id": ["a", "b", "c"]})
    pred = pd.DataFrame({"video_id": "v", "track_id": [1, 2, 3], "player_id": ["a", "x", ""],
                         "identity_confidence": [0.95, 0.97, None]})
    r = identity_metrics(pred, truth)
    assert (r["correct"], r["incorrect"], r["unknown"], r["high_conf_wrong"]) == (1, 1, 1, 1)
