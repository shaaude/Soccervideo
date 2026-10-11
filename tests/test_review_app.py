"""The Streamlit reviewer renders, saves a review with an audit record, and refuses invalid confirmations."""
import shutil

import pandas as pd
import pytest

from conftest import REPO

st_testing = pytest.importorskip("streamlit.testing.v1")


@pytest.fixture
def workdir(tmp_path, synthetic_video, monkeypatch):
    (tmp_path / "data/normalized").mkdir(parents=True)
    (tmp_path / "data/processed/match001").mkdir(parents=True)
    shutil.copy(synthetic_video, tmp_path / "data/normalized/match001-test.mp4")
    pd.DataFrame({"event_id": ["match001-transfer-000001"], "match_id": "match001",
                  "event_type": "ball_transfer_candidate", "video_time_s": [3.0], "initiator_track_id": [4],
                  "receiver_track_id": [9], "rule_version": "contact_v2"}).to_csv(
        tmp_path / "data/processed/match001/event_candidates.csv", index=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_review_app_save_and_guardrails(workdir):
    at = st_testing.AppTest.from_file(str(REPO / "app/review.py"), default_timeout=60).run()
    assert not at.exception
    # Confirming the machine-only candidate type is refused
    form_status = [s for s in at.selectbox if s.label == "Review status"][0]
    form_status.set_value("verified")
    [t for t in at.text_input if t.label == "Reviewer ID"][0].set_value("rev1")
    [b for b in at.button if b.label == "Save review"][0].click().run()
    assert any("machine-only" in e.value for e in at.error)
    # Choose a real type, actor, and save
    [s for s in at.selectbox if s.label == "Observed event type"][0].set_value("pass").run()
    [s for s in at.selectbox if s.label == "Outcome"][0].set_value("completed")
    [t for t in at.text_input if t.label.startswith("Actor roster")][0].set_value("p12")
    [s for s in at.selectbox if s.label == "Review status"][0].set_value("verified")
    [t for t in at.text_input if t.label == "Reviewer ID"][0].set_value("rev1")
    [b for b in at.button if b.label == "Save review"][0].click().run()
    assert not at.exception
    saved = pd.read_csv(workdir / "data/annotations/match001/reviewed_events.csv", dtype=str)
    row = saved.iloc[0]
    assert (row.event_type, row.actor_player_id, row.review_status, row.reviewer_id) == ("pass", "p12", "verified", "rev1")
    assert (workdir / "data/annotations/match001/review_audit.jsonl").read_text().count("\n") == 1
