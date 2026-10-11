import pandas as pd
from sqlalchemy.orm import Session

from soccer_vision.db.importer import import_reviews
from soccer_vision.db.models import Base, Event, EventReview, Match, MatchVideo, Player, PlayerAppearance, Team
from soccer_vision.db.session import make_engine
from soccer_vision.review.store import ReviewStore


def test_file_reviews_apply_once_and_never_guess_players(tmp_path):
    eng = make_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    Base.metadata.create_all(eng)
    with Session(eng) as s, s.begin():
        s.add_all([Team(team_id="A", team_name="A"), Player(player_id="p1", display_name="P1")])
        s.flush()
        s.add(Match(match_id="m1"))
        s.flush()
        s.add_all([MatchVideo(video_id="v1", match_id="m1", file_path="x.mp4", sha256="0", fps=25),
                   PlayerAppearance(match_id="m1", team_id="A", player_id="p1")])
        s.flush()
        for i in (1, 2):
            s.add(Event(event_id=f"m1-transfer-00000{i}", match_id="m1", video_id="v1", video_time_seconds=i,
                        event_type="ball_transfer_candidate", original_prediction={}))
    store = ReviewStore(tmp_path / "ann", "m1")
    store.sync_candidates(pd.DataFrame({"event_id": ["m1-transfer-000001", "m1-transfer-000002"], "match_id": "m1",
                                        "event_type": "ball_transfer_candidate", "video_time_s": [1.0, 2.0]}))
    store.save_review("m1-transfer-000001", {"event_type": "pass", "outcome": "completed", "review_status": "verified",
                                            "actor_player_id": "p1"}, "rev", 0)
    store.save_review("m1-transfer-000002", {"event_type": "pass", "outcome": "completed", "review_status": "verified",
                                            "actor_player_id": "p99"}, "rev", 0)
    r = import_reviews(eng, tmp_path / "ann", "m1")
    assert r["applied"] == 1 and len(r["skipped"]) == 1 and "p99" in r["skipped"][0]
    assert import_reviews(eng, tmp_path / "ann", "m1")["applied"] == 0
    with Session(eng) as s:
        e = s.get(Event, "m1-transfer-000001")
        assert (e.event_type, e.review_status, e.version) == ("pass", "verified", 1) and e.actor_appearance_id
        assert s.get(Event, "m1-transfer-000002").review_status == "unreviewed"
        assert s.query(EventReview).count() == 1
