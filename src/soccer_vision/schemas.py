"""Column contracts for every file-based artifact, plus validators.

Canonical person/ball class IDs used in all exported CSVs (independent of the
detector's own numbering; each detector is mapped onto these via a class map):
    0 ball, 1 goalkeeper, 2 player, 3 referee
"""
from __future__ import annotations

import pandas as pd
import numpy as np

CLASS_BALL, CLASS_GOALKEEPER, CLASS_PLAYER, CLASS_REFEREE = 0, 1, 2, 3
CANONICAL_CLASSES = {
    "ball": CLASS_BALL,
    "goalkeeper": CLASS_GOALKEEPER,
    "player": CLASS_PLAYER,
    "referee": CLASS_REFEREE,
}
PERSON_CLASSES = (CLASS_GOALKEEPER, CLASS_PLAYER, CLASS_REFEREE)
OUTFIELD_ACTOR_CLASSES = (CLASS_GOALKEEPER, CLASS_PLAYER)

# Guide Step 6 columns first (unchanged order), then run-scoping columns (Step 21.3).
TRACK_COLUMNS = [
    "match_id", "frame_index", "time_s", "track_id", "class_id", "confidence",
    "x1", "y1", "x2", "y2", "foot_x", "foot_y",
    "video_id", "run_id", "segment_id", "class_name",
]
BALL_COLUMNS = [
    "match_id", "frame_index", "time_s", "confidence",
    "x1", "y1", "x2", "y2", "center_x", "center_y",
    "video_id", "run_id", "rank_in_frame",
]

TRACK_DTYPES = {
    "match_id": str, "frame_index": "int64", "time_s": "float64", "track_id": "Int64",
    "class_id": "int64", "confidence": "float64", "video_id": str, "run_id": str,
    "segment_id": "int64", "class_name": str,
}
BALL_DTYPES = {"match_id": str, "frame_index": "int64", "time_s": "float64",
               "confidence": "float64", "video_id": str, "run_id": str, "rank_in_frame": "int64"}

CONTACT_COLUMNS = ["match_id", "frame_index", "time_s", "raw_track_id", "supported_track_id",
                   "distance_over_height", "ball_confidence"]
EVENT_CANDIDATE_COLUMNS = [
    "event_id", "match_id", "event_type", "video_time_s", "receiver_time_s",
    "initiator_track_id", "receiver_track_id", "gap_seconds", "review_status", "rule_version",
    "initiator_support_frames", "receiver_support_frames", "min_distance_over_height",
    "video_id", "run_id",
]
SUPPRESSED_COLUMNS = ["match_id", "prev_track_id", "next_track_id", "prev_end_time_s",
                      "next_start_time_s", "gap_seconds", "reason"]

# Human-review store (Step 12). Predicted fields are preserved alongside corrections.
REVIEW_COLUMNS = [
    "event_id", "source", "match_id", "event_type", "outcome",
    "video_time_s", "match_clock_s", "actor_track_id", "actor_player_id",
    "receiver_track_id", "receiver_player_id", "review_status", "review_notes",
    "reviewer_id", "reviewed_at", "version",
    "predicted_event_type", "predicted_video_time_s", "predicted_initiator_track_id",
    "predicted_receiver_track_id", "rule_version", "orphaned",
]

UNKNOWN_PLAYER = "UNKNOWN"  # explicit reviewer statement: event happened, actor not identifiable


class SchemaError(ValueError):
    pass


def require_columns(df: pd.DataFrame, cols, label: str) -> None:
    missing = set(cols).difference(df.columns)
    if missing:
        raise SchemaError(f"{label} missing columns: {sorted(missing)}")


def _check_common(df: pd.DataFrame, label: str, match_id: str | None) -> list[str]:
    errors = []
    if df.empty:
        return errors
    if df["match_id"].isna().any():
        errors.append(f"{label}: missing match_id")
    elif match_id is not None and set(df["match_id"].astype(str)) != {match_id}:
        errors.append(f"{label}: match_id values {sorted(set(df['match_id'].astype(str)))} != {match_id}")
    fi = df["frame_index"]
    if fi.isna().any() or (fi < 0).any() or not np.all(np.mod(fi.astype(float), 1) == 0):
        errors.append(f"{label}: frame_index must be non-negative integers")
    elif not fi.is_monotonic_increasing:
        errors.append(f"{label}: rows must be grouped in increasing frame_index order")
    conf = df["confidence"]
    if conf.isna().any() or not conf.between(0, 1).all():
        errors.append(f"{label}: confidence outside [0,1]")
    boxes = df[["x1", "y1", "x2", "y2"]]
    if boxes.isna().any().any():
        errors.append(f"{label}: NaN box coordinate")
    elif ((df["x2"] < df["x1"]) | (df["y2"] < df["y1"])).any():
        errors.append(f"{label}: box with x2<x1 or y2<y1")
    if "time_s" in df and df["time_s"].notna().all() and not df["time_s"].is_monotonic_increasing:
        errors.append(f"{label}: time_s not monotonically increasing")
    return errors


def validate_tracks(df: pd.DataFrame, match_id: str | None = None, fps: float | None = None) -> list[str]:
    require_columns(df, TRACK_COLUMNS[:12], "tracks")
    errors = _check_common(df, "tracks", match_id)
    if df.empty:
        return errors
    if df["track_id"].isna().any():
        errors.append("tracks: track_id missing (untracked detections must not be exported here)")
    bad_cls = set(df["class_id"].unique()).difference(PERSON_CLASSES)
    if bad_cls:
        errors.append(f"tracks: non-person canonical class_id(s) {sorted(bad_cls)}")
    if not np.allclose(df["foot_x"], (df["x1"] + df["x2"]) / 2, rtol=0, atol=0.01) or \
            not np.allclose(df["foot_y"], df["y2"], rtol=0, atol=0.01):
        errors.append("tracks: foot point must be bottom-centre of the box")
    if fps:
        if not np.allclose(df["time_s"], df["frame_index"] / fps, atol=1e-3):
            errors.append("tracks: time_s != frame_index / fps")
    key = ["frame_index", "track_id"] + (["run_id"] if "run_id" in df else [])
    if df.duplicated(key).any():
        errors.append("tracks: duplicate (frame_index, track_id)")
    return errors


def validate_balls(df: pd.DataFrame, match_id: str | None = None, fps: float | None = None) -> list[str]:
    require_columns(df, BALL_COLUMNS[:10], "balls")
    errors = _check_common(df, "balls", match_id)
    if df.empty:
        return errors
    if not np.allclose(df["center_x"], (df["x1"] + df["x2"]) / 2, rtol=0, atol=0.01) or \
       not np.allclose(df["center_y"], (df["y1"] + df["y2"]) / 2, rtol=0, atol=0.01):
        errors.append("balls: center must be box centre")
    if fps and not np.allclose(df["time_s"], df["frame_index"] / fps, atol=1e-3):
        errors.append("balls: time_s != frame_index / fps")
    return errors


def read_tracks(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={k: v for k, v in TRACK_DTYPES.items() if v is str})


def read_balls(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={k: v for k, v in BALL_DTYPES.items() if v is str})


def read_review_csv(path) -> pd.DataFrame:
    """Read a review store/export with identifier columns kept as strings (no 12 -> 12.0)."""
    str_cols = {c: str for c in ("event_id", "match_id", "actor_player_id", "receiver_player_id",
                                 "event_type", "outcome", "review_status", "reviewer_id",
                                 "review_notes", "source", "rule_version")}
    df = pd.read_csv(path, dtype=str_cols)
    for c in str_cols:
        if c in df:
            df[c] = df[c].fillna("")
    return df
