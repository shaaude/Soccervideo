import numpy as np
import pandas as pd

from soccer_vision.analytics.movement import ball_zones, player_distances, track_distances
from soccer_vision.report import event_counts, pitch_zones_svg


def _track(tid, n=250, fps=25, speed=2.0, jump_at=None):
    x = np.arange(n) * speed / fps + 10
    if jump_at:
        x[jump_at:] += 30          # an ID swap: 30 m teleport
    return pd.DataFrame({"track_id": tid, "frame_index": np.arange(n), "class_id": 2, "field_x_m": x,
                         "field_y_m": 20.0, "calibrated": True, "off_pitch": False})


def test_distance_while_tracked_and_teleports_discarded():
    d = track_distances(pd.concat([_track(1), _track(2, jump_at=100), _track(3, n=40)]), fps=25)
    r = d.set_index("track_id")
    assert abs(r.loc[1, "distance_m"] - 19.9) < 0.3 and abs(r.loc[1, "top_speed_ms"] - 2.0) < 0.05
    assert abs(r.loc[2, "distance_m"] - 19.8) < 0.5 and r.loc[2, "discarded_steps"] >= 1   # jump not counted
    assert 3 not in r.index                                     # under 3 s tracked: not reported


def test_distances_only_for_confirmed_identities():
    per_track = pd.DataFrame({"track_id": [1, 2, 3], "tracked_seconds": [10, 10, 10], "distance_m": [20, 30, 40]})
    a = pd.DataFrame({"track_id": [1, 2, 3], "player_id": ["p7", "p7", "p9"],
                      "review_status": ["verified", "corrected", "unreviewed"]})
    out = player_distances(per_track, a).set_index("player_id")
    assert out.loc["p7", "distance_m"] == 50 and out.loc["p7", "tracklets"] == 2 and "p9" not in out.index


def test_ball_zones_and_pitch_svg():
    xy = np.array([[5, 5], [5, 60], [100, 34], [100, 34]], float)
    g = ball_zones(xy, 105, 68)
    assert g.shape == (3, 6) and abs(g.sum() - 1) < 1e-9 and g[1, 5] == 0.5
    svg = pitch_zones_svg(g, 105, 68)
    assert "50%" in svg and "<svg" in svg


def test_event_counts_use_confirmed_only():
    r = pd.DataFrame({"event_type": ["shot", "shot", "corner", "tackle", "tackle"],
                      "outcome": ["goal", "missed", "not_applicable", "won", "lost"],
                      "review_status": ["verified", "unreviewed", "corrected", "verified", "verified"]})
    c = dict(event_counts(r))
    assert c["Shots"] == 1 and c["Goals (from shots)"] == 1 and c["Corners"] == 1
    assert c["Battles won (tackles)"] == 1 and c["Battles lost (tackles)"] == 1
