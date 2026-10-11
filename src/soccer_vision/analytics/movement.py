"""Pitch-based movement and ball-location analytics. Needs a calibrated run (tracks_field.parquet).

Distances are only computed in metres from calibrated frames. Tracking gaps are never bridged, and steps
that imply impossible speeds (tracker ID swaps, projection glitches) are discarded and counted, so
"distance while tracked" is a lower bound rather than a guess.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_SPEED_MS = 10.0          # faster than any youth or adult sprint: treat as a tracking error
SMOOTH_S = 0.4               # rolling median window to suppress box jitter
MAX_GAP_S = 0.5              # do not join positions across longer gaps


def track_distances(field_tracks: pd.DataFrame, fps: float, min_tracked_s: float = 3.0) -> pd.DataFrame:
    off = field_tracks["off_pitch"] if "off_pitch" in field_tracks else pd.Series(False, index=field_tracks.index)
    t = field_tracks[field_tracks["calibrated"].astype(bool) & ~off.astype(bool)
                     & field_tracks["class_id"].isin([1, 2])].copy()
    rows = []
    win = max(1, int(round(SMOOTH_S * fps)))
    for tid, g in t.groupby("track_id"):
        g = g.sort_values("frame_index")
        x = g["field_x_m"].rolling(win, center=True, min_periods=1).median().to_numpy()
        y = g["field_y_m"].rolling(win, center=True, min_periods=1).median().to_numpy()
        f = g["frame_index"].to_numpy()
        dt = np.diff(f) / fps
        step = np.hypot(np.diff(x), np.diff(y))
        ok = (dt > 0) & (dt <= MAX_GAP_S)
        speed = np.where(ok, step / np.where(dt > 0, dt, 1), np.nan)
        good = ok & (speed <= MAX_SPEED_MS)
        tracked_s = float(dt[ok].sum())
        if tracked_s < min_tracked_s:
            continue
        sp = speed[good]
        rows.append({"track_id": int(tid), "tracked_seconds": round(tracked_s, 1),
                     "distance_m": round(float(step[good].sum()), 1),
                     "top_speed_ms": round(float(np.percentile(sp, 95)), 2) if len(sp) else None,
                     "discarded_steps": int((ok & ~good).sum()),
                     "start_s": round(f[0] / fps, 1), "end_s": round(f[-1] / fps, 1)})
    return pd.DataFrame(rows, columns=["track_id", "tracked_seconds", "distance_m", "top_speed_ms",
                                       "discarded_steps", "start_s", "end_s"])


def player_distances(per_track: pd.DataFrame, assignments: pd.DataFrame | None) -> pd.DataFrame:
    """Sum tracklet distances per roster player, using human-confirmed identities only."""
    if assignments is None or per_track.empty:
        return pd.DataFrame(columns=["player_id", "tracklets", "tracked_seconds", "distance_m"])
    ok = assignments[assignments["review_status"].isin(["verified", "corrected"])
                     & assignments["player_id"].astype(str).str.strip().ne("")
                     & assignments["player_id"].astype(str).str.upper().ne("UNKNOWN")]
    m = per_track.merge(ok[["track_id", "player_id"]].astype({"track_id": int, "player_id": str}), on="track_id")
    out = m.groupby("player_id").agg(tracklets=("track_id", "nunique"), tracked_seconds=("tracked_seconds", "sum"),
                                     distance_m=("distance_m", "sum")).reset_index()
    return out.round(1).sort_values("distance_m", ascending=False)


def ball_zones(ball_xy: np.ndarray, length_m: float, width_m: float, nx: int = 6, ny: int = 3) -> np.ndarray:
    """Share of ball observations per pitch zone (nx along the length, ny across). Rows = ny, cols = nx."""
    grid = np.zeros((ny, nx), float)
    if len(ball_xy) == 0:
        return grid
    xs = np.clip((ball_xy[:, 0] / length_m * nx).astype(int), 0, nx - 1)
    ys = np.clip((ball_xy[:, 1] / width_m * ny).astype(int), 0, ny - 1)
    np.add.at(grid, (ys, xs), 1)
    return grid / grid.sum()


def project_ball_ground(balls: pd.DataFrame, transforms: dict, conf: float = 0.15) -> np.ndarray:
    """Approximate ball ground position from the bottom of its box (best ball per frame, calibrated frames).
    Airborne balls project too far away; zone-level summaries tolerate this, metre-level claims do not."""
    from ..calibration.homography import apply_mapping
    b = balls[balls["confidence"] >= conf].sort_values(["frame_index", "confidence"], ascending=[True, False])
    b = b.drop_duplicates("frame_index")
    L, W = transforms["field_length_m"], transforms["field_width_m"]
    out = []
    for seg in transforms["segments"]:
        mapping = seg.get("mapping") or ({"type": "homography", "H": seg["H"]} if seg.get("H") else None)
        if mapping is None or seg["status"] not in ("calibrated", "calibrated_unvalidated"):
            continue
        s = b[b["frame_index"].between(seg["start_frame"], seg["end_frame"])]
        if s.empty:
            continue
        xy = apply_mapping(mapping, np.stack([s["center_x"], s["y2"]], axis=1))
        keep = (xy[:, 0] >= -2) & (xy[:, 0] <= L + 2) & (xy[:, 1] >= -2) & (xy[:, 1] <= W + 2)
        out.append(xy[keep])
    return np.vstack(out) if out else np.zeros((0, 2))
