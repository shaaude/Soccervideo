import pytest

from soccer_vision.events.candidates import CandidateParams, generate, generate_with_audit, load_inputs
from conftest import make_tracks_balls


def owner_pass(f):
    # player 1 holds 0-9, ball in flight 10-14 (0.2 s), player 2 holds 15-29
    return 1 if f < 10 else ("air" if f < 15 else 2)


def test_transfer_is_only_candidate():
    tracks, balls = make_tracks_balls(owner_pass, 30)
    contacts, events = generate(tracks, balls, "m1", fps=25)
    assert len(events) == 1
    e = events.iloc[0]
    assert (e.initiator_track_id, e.receiver_track_id) == (1, 2)
    assert e.event_type == "ball_transfer_candidate" and e.review_status == "unreviewed"
    assert len(contacts) == 30  # one row per observed ball frame


def test_receiver_time_is_first_touch_not_delayed_by_support_window():
    tracks, balls = make_tracks_balls(owner_pass, 30)
    _, events = generate(tracks, balls, "m1", fps=25)
    assert events.iloc[0].receiver_time_s == pytest.approx(15 / 25)
    assert events.iloc[0].video_time_s == pytest.approx(9 / 25)
    assert events.iloc[0].gap_seconds == pytest.approx(6 / 25)


def test_teleport_below_min_gap_is_suppressed_with_reason():
    tracks, balls = make_tracks_balls(lambda f: 1 if f < 10 else 2, 20)
    _, events, suppressed = generate_with_audit(tracks, balls, "m1", CandidateParams(fps=25))
    assert events.empty
    assert suppressed.iloc[0].reason == "gap_below_min"


def test_long_gap_suppressed():
    tracks, balls = make_tracks_balls(lambda f: 1 if f < 10 else (None if f < 135 else 2), 150)
    _, events, suppressed = generate_with_audit(tracks, balls, "m1", CandidateParams(fps=25))
    assert events.empty and list(suppressed.reason) == ["gap_above_max"]


def test_no_transfer_for_same_player():
    tracks, balls = make_tracks_balls(lambda f: 1, 20)
    _, events = generate(tracks, balls, "m1", fps=25)
    assert events.empty


def test_same_player_regaining_after_flight_is_suppressed_not_emitted():
    tracks, balls = make_tracks_balls(lambda f: 1 if f < 10 or f >= 25 else "air", 35)
    _, events, suppressed = generate_with_audit(tracks, balls, "m1", CandidateParams(fps=25))
    assert events.empty and list(suppressed.reason) == ["same_track_regained"]


def test_ambiguous_contact_is_left_unknown():
    # two players equally close to the ball: no contact is attributed
    tracks, balls = make_tracks_balls(lambda f: 1, 20, players=((1, 100), (2, 110)))
    balls["center_x"] = 105
    balls["x1"], balls["x2"] = 102, 108
    contacts, events = generate(tracks, balls, "m1", fps=25)
    assert contacts["raw_track_id"].isna().all() and events.empty


def test_referee_is_never_an_actor():
    tracks, balls = make_tracks_balls(owner_pass, 30)
    tracks.loc[tracks.track_id == 2, "class_id"] = 3
    _, events = generate(tracks, balls, "m1", fps=25)
    assert events.empty


def test_missing_ball_produces_no_events():
    tracks, balls = make_tracks_balls(owner_pass, 30)
    contacts, events = generate(tracks, balls.iloc[:0], "m1", fps=25)
    assert events.empty and contacts.empty and list(events.columns)


def test_invalid_fps():
    tracks, balls = make_tracks_balls(owner_pass, 30)
    with pytest.raises(ValueError):
        generate(tracks, balls, "m1", fps=0)


def test_load_inputs_rejects_foreign_match(tmp_path):
    tracks, balls = make_tracks_balls(owner_pass, 30)
    tracks.to_csv(tmp_path / "t.csv", index=False)
    balls.assign(match_id="other").to_csv(tmp_path / "b.csv", index=False)
    with pytest.raises(ValueError, match="different"):
        load_inputs(tmp_path / "t.csv", tmp_path / "b.csv", "m1")
