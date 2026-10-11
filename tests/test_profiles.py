import io

import pandas as pd
import pytest

from soccer_vision.analytics.profiles import ProfileError, build_profiles, profiles_from_events
from soccer_vision.schemas import read_review_csv


def events():
    return pd.DataFrame([
        dict(event_id="e1", match_id="m1", event_type="pass", outcome="completed", actor_player_id="p1", review_status="verified"),
        dict(event_id="e2", match_id="m1", event_type="pass", outcome="incomplete", actor_player_id="p1", review_status="corrected"),
        dict(event_id="e3", match_id="m1", event_type="goal", outcome="scored", actor_player_id="p1", review_status="unreviewed"),
        dict(event_id="e4", match_id="m2", event_type="pass", outcome="unknown", actor_player_id="p1", review_status="verified"),
    ])


def test_verified_only_and_unknown_denominator():
    r = profiles_from_events(events()).iloc[0]
    assert r.total_reviewed_events == 3
    assert r.pass_attempts_resolved == 2 and r.pass_outcome_unknown == 1
    assert r.passing_accuracy_pct == 50.0
    assert r.goals == 0
    assert r.matches_with_reviewed_events == 2
    assert r.shots_per90 is None  # no exposure supplied -> no rates


def test_reject_duplicate_event_id():
    e = events()
    e.loc[1, "event_id"] = "e1"
    with pytest.raises(ProfileError):
        profiles_from_events(e)


def test_reject_verified_missing_actor():
    e = events()
    e.loc[0, "actor_player_id"] = None
    with pytest.raises(ProfileError, match="lack actor"):
        profiles_from_events(e)


def test_unknown_actor_is_excluded_and_reported():
    e = events()
    e.loc[0, "actor_player_id"] = "UNKNOWN"
    prof, rep = build_profiles(e)
    assert prof.iloc[0].completed_passes == 0
    assert rep["confirmed_unknown_actor"] == 1


def test_team_level_event_without_actor_does_not_block():
    e = pd.concat([events(), pd.DataFrame([dict(event_id="e5", match_id="m1", event_type="ball_out",
                                                outcome="not_applicable", actor_player_id="", review_status="verified")])])
    prof, rep = build_profiles(e)
    assert rep["team_level_confirmed"] == 1 and len(prof) == 1


def test_invalid_outcome_on_confirmed_event_rejected():
    e = events()
    e.loc[0, "outcome"] = "scored"
    with pytest.raises(ProfileError, match="not valid"):
        profiles_from_events(e)


def test_unreviewed_goal_never_counted_and_shot_goal_counted_once():
    e = pd.concat([events(), pd.DataFrame([
        dict(event_id="s1", match_id="m1", event_type="shot", outcome="goal", actor_player_id="p1", review_status="verified"),
        dict(event_id="g1", match_id="m1", event_type="goal", outcome="not_applicable", actor_player_id="", review_status="verified"),
        dict(event_id="s2", match_id="m1", event_type="shot", outcome="goal", actor_player_id="p1", review_status="unreviewed"),
    ])])
    r = profiles_from_events(e).iloc[0]
    assert (r.shots, r.goals, r.shots_on_target) == (1, 1, 1)


def test_machine_candidate_cannot_be_confirmed():
    e = events()
    e.loc[0, "event_type"] = "ball_transfer_candidate"
    e.loc[0, "outcome"] = "unknown"
    with pytest.raises(ProfileError, match="machine-only"):
        profiles_from_events(e)


def test_numeric_player_ids_stay_strings_when_read_from_csv():
    csv = ("event_id,match_id,event_type,outcome,actor_player_id,review_status\n"
           "e1,m,pass,completed,12,verified\ne2,m,ball_transfer_candidate,unknown,,unreviewed\n")
    prof = profiles_from_events(read_review_csv(io.StringIO(csv)))
    assert prof.player_id.tolist() == ["12"]


def test_reviews_from_superseded_run_are_not_double_counted():
    e = events().assign(orphaned="")
    e.loc[1, "orphaned"] = "true"
    prof, rep = build_profiles(e)
    assert prof.iloc[0].pass_attempts_resolved == 1 and rep["excluded_confirmed_from_superseded_runs"] == 1


def test_per90_only_with_exposure():
    exp = pd.DataFrame([{"match_id": "m1", "player_id": "p1", "reviewed_minutes": 45.0},
                        {"match_id": "m2", "player_id": "p1", "reviewed_minutes": 45.0}])
    r = profiles_from_events(events(), exposure=exp).iloc[0]
    assert r.reviewed_minutes == 90.0 and r.passes_completed_per90 == 1.0
