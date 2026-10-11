"""Clean ball candidates before they are used for events, maps or statistics.

Two kinds of false "balls" are common on youth-field footage:
  * static objects (a cone, a logo, a bag at the frame edge) detected at the same pixel for 20 s or more
    (long enough that a ball waiting for a free kick or corner is not mistaken for one);
  * detections well outside the pitch (sky, trees, stands).
Removed candidates are counted, never silently dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def best_per_frame(balls: pd.DataFrame, conf: float) -> pd.DataFrame:
    b = balls[balls["confidence"] >= conf].sort_values(["frame_index", "confidence"], ascending=[True, False])
    return b.drop_duplicates("frame_index")


def static_mask(balls: pd.DataFrame, fps: float, cell_px: float = 6.0, min_seconds: float = 20.0,
                min_share: float = 0.5) -> np.ndarray:
    """True for detections at a pixel location that recurs over >= min_seconds of video and accounts for
    >= min_share of the frames in that span (a ball does not stay put that long while play is on)."""
    if balls.empty:
        return np.zeros(0, bool)
    key = (np.round(balls["center_x"].to_numpy() / cell_px).astype(int) * 100003
           + np.round(balls["center_y"].to_numpy() / cell_px).astype(int))
    out = np.zeros(len(balls), bool)
    frames = balls["frame_index"].to_numpy()
    for k in np.unique(key):
        idx = np.flatnonzero(key == k)
        f = frames[idx]
        span = (f.max() - f.min() + 1) / fps
        if span >= min_seconds and len(np.unique(f)) >= min_share * min_seconds * fps:
            out[idx] = True
    return out


def roi_mask(balls: pd.DataFrame, roi_polygon, margin_px: float = 40.0) -> np.ndarray:
    if not roi_polygon or balls.empty:
        return np.ones(len(balls), bool)
    import cv2
    poly = np.asarray(roi_polygon, np.float32).reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= -margin_px
                     for x, y in zip(balls["center_x"], balls["center_y"])], bool)


def clean_balls(balls: pd.DataFrame, fps: float, roi_polygon=None) -> tuple[pd.DataFrame, dict]:
    if balls.empty:
        return balls, {"static_removed": 0, "outside_pitch_removed": 0}
    st = static_mask(balls, fps)
    inside = roi_mask(balls, roi_polygon)
    keep = ~st & inside
    return balls[keep].copy(), {"static_removed": int(st.sum()), "outside_pitch_removed": int((~inside & ~st).sum())}
