"""Step 10: conservative contact episodes and A->B ball-transfer CANDIDATES.

This stage never creates passes, possessions, or player statistics. Every output row is an
unreviewed hypothesis with the evidence needed to review it. Suppressed transfers are kept,
with a reason, in suppressed_candidates.csv (Step 10.5).

Changes from kit rule contact_v1 (rule_version contact_v2):
  * minimum transfer gap (guide: 0.12-4.0 s) is enforced
  * episodes start at the first raw contact, so receiver_time_s is no longer delayed by the
    support window
  * suppression reasons are written out instead of silently dropped
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from ..schemas import (CONTACT_COLUMNS, EVENT_CANDIDATE_COLUMNS, OUTFIELD_ACTOR_CLASSES,
                       SUPPRESSED_COLUMNS, require_columns)

RULE_VERSION = "contact_v2"
TRACK_REQUIRED = ["match_id", "frame_index", "track_id", "class_id", "confidence",
                  "x1", "y1", "x2", "y2", "foot_x", "foot_y"]
BALL_REQUIRED = ["match_id", "frame_index", "confidence", "center_x", "center_y"]


@dataclass
class CandidateParams:
    fps: float = 25.0
    ball_threshold: float = 0.15
    player_threshold: float = 0.25
    max_distance: float = 0.45       # player heights between foot point and ball centre
    margin: float = 0.08             # nearest must beat runner-up by this much, else ambiguous
    support_window: int = 7
    support_min: int = 3
    segment_gap_s: float = 0.35
    min_transfer_gap_s: float = 0.12
    max_transfer_gap_s: float = 4.0
    min_box_height_px: float = 3.0

    def check(self):
        if self.fps <= 0 or self.support_window < 1 or not 1 <= self.support_min <= self.support_window:
            raise ValueError("Invalid FPS / rolling window configuration")
        if not 0 <= self.min_transfer_gap_s <= self.max_transfer_gap_s:
            raise ValueError("Invalid transfer gap bounds")


def load_inputs(tracks_path, balls_path, match_id):
    t = pd.read_csv(tracks_path, dtype={"match_id": str})
    b = pd.read_csv(balls_path, dtype={"match_id": str})
    require_columns(t, TRACK_REQUIRED, "tracks")
    require_columns(b, BALL_REQUIRED, "balls")
    errs = []
    if not t.empty:
        errs += [e for e in _light_checks(t, "tracks", match_id)]
    if not b.empty:
        errs += [e for e in _light_checks(b, "balls", match_id)]
    if errs:
        raise ValueError("; ".join(errs))
    return t, b


def _light_checks(df, label, match_id):
    errs = []
    if df["match_id"].isna().any():
        errs.append(f"{label}: missing match_id")
    elif set(df["match_id"].astype(str)) != {match_id}:
        errs.append(f"Inputs contain match IDs different from --match-id ({label})")
    if df["frame_index"].isna().any() or (df["frame_index"] < 0).any():
        errs.append(f"{label} contains invalid frame indices")
    if not df["confidence"].between(0, 1).all():
        errs.append(f"{label} contains invalid confidences")
    return errs


def _raw_contacts(tracks, balls, p: CandidateParams):
    players_by_frame = {int(k): v for k, v in tracks.groupby("frame_index", sort=False)}
    rows = []
    for ball in balls.itertuples(index=False):
        frame = int(ball.frame_index)
        people = players_by_frame.get(frame)
        raw_track, distance = None, None
        if people is not None:
            height = (people["y2"] - people["y1"]).to_numpy(dtype=float)
            dist = np.hypot(people["foot_x"].to_numpy(float) - float(ball.center_x),
                            people["foot_y"].to_numpy(float) - float(ball.center_y)) / np.maximum(height, 1.0)
            allowed = (height >= p.min_box_height_px) & np.isfinite(dist) & (dist <= p.max_distance)
            ranking = np.flatnonzero(allowed)
            if len(ranking):
                ranking = ranking[np.argsort(dist[ranking])]
                nearest = ranking[0]
                if len(ranking) == 1 or dist[ranking[1]] - dist[nearest] >= p.margin:
                    raw_track = int(people.iloc[nearest]["track_id"])
                    distance = float(dist[nearest])
        rows.append({"frame_index": frame, "raw_track_id": raw_track, "distance_over_height": distance,
                     "ball_confidence": float(ball.confidence)})
    return pd.DataFrame(rows, columns=["frame_index", "raw_track_id", "distance_over_height", "ball_confidence"])


def _supported(frames: list[int], window: int, need: int) -> bool:
    f = np.asarray(frames)
    for i in range(len(f)):
        if np.searchsorted(f, f[i] + window - 1, side="right") - i >= need:
            return True
    return False


def _episodes(raw: pd.DataFrame, p: CandidateParams):
    """Runs of raw contacts by one track; a different track or a long gap starts a new run."""
    hits = raw[raw["raw_track_id"].notna()]
    runs = []
    for r in hits.itertuples(index=False):
        tid, f = int(r.raw_track_id), int(r.frame_index)
        if runs and runs[-1]["track_id"] == tid and (f - runs[-1]["frames"][-1]) / p.fps <= p.segment_gap_s:
            runs[-1]["frames"].append(f)
            runs[-1]["dists"].append(r.distance_over_height)
        else:
            runs.append({"track_id": tid, "frames": [f], "dists": [r.distance_over_height]})
    valid = [e for e in runs if len(e["frames"]) >= 2 and _supported(e["frames"], p.support_window, p.support_min)]
    merged = []
    for e in valid:  # a rejected blip by another track between two runs of the same track is ignored
        if merged and merged[-1]["track_id"] == e["track_id"] and \
                (e["frames"][0] - merged[-1]["frames"][-1]) / p.fps <= p.segment_gap_s:
            merged[-1]["frames"] += e["frames"]
            merged[-1]["dists"] += e["dists"]
        else:
            merged.append({"track_id": e["track_id"], "frames": list(e["frames"]), "dists": list(e["dists"])})
    for e in merged:
        e["start_frame"], e["end_frame"] = e["frames"][0], e["frames"][-1]
    return merged


def generate_with_audit(tracks: pd.DataFrame, balls: pd.DataFrame, match_id: str, params: CandidateParams | None = None,
                        video_id: str = "", run_id: str = ""):
    p = params or CandidateParams()
    p.check()
    empty = (pd.DataFrame(columns=CONTACT_COLUMNS), pd.DataFrame(columns=EVENT_CANDIDATE_COLUMNS),
             pd.DataFrame(columns=SUPPRESSED_COLUMNS))
    if tracks.empty or balls.empty:
        return empty
    tracks = tracks[tracks["class_id"].isin(OUTFIELD_ACTOR_CLASSES) & (tracks["confidence"] >= p.player_threshold)
                    & tracks["track_id"].notna()].copy()
    balls = balls[balls["confidence"] >= p.ball_threshold].copy()
    if balls.empty:
        return empty
    # One provisional ball location per observed frame (highest confidence). Not ball identity tracking.
    balls = balls.sort_values(["frame_index", "confidence"], ascending=[True, False]).drop_duplicates("frame_index")
    raw = _raw_contacts(tracks, balls, p)
    episodes = _episodes(raw, p)

    supported = {}
    for e in episodes:
        for f in range(e["start_frame"], e["end_frame"] + 1):
            supported[f] = e["track_id"]
    contacts = raw.assign(match_id=match_id, time_s=raw["frame_index"] / p.fps,
                          supported_track_id=raw["frame_index"].map(supported))
    contacts = contacts[CONTACT_COLUMNS]

    events, suppressed = [], []
    for prev, nxt in zip(episodes, episodes[1:]):
        gap = (nxt["start_frame"] - prev["end_frame"]) / p.fps
        reason = None
        if prev["track_id"] == nxt["track_id"]:
            reason = "same_track_regained"
        elif gap < p.min_transfer_gap_s:
            reason = "gap_below_min"
        elif gap > p.max_transfer_gap_s:
            reason = "gap_above_max"
        if reason:
            suppressed.append({"match_id": match_id, "prev_track_id": prev["track_id"], "next_track_id": nxt["track_id"],
                               "prev_end_time_s": round(prev["end_frame"] / p.fps, 3),
                               "next_start_time_s": round(nxt["start_frame"] / p.fps, 3),
                               "gap_seconds": round(gap, 3), "reason": reason})
            continue
        events.append({
            # Track IDs are run-scoped, so candidate IDs from a pipeline run carry the run prefix:
            # a re-run can never silently re-attach an old review to a different event.
            "event_id": (f"{match_id}-{run_id[:8]}-transfer-{len(events) + 1:06d}" if run_id
                         else f"{match_id}-transfer-{len(events) + 1:06d}"),
            "match_id": match_id,
            "event_type": "ball_transfer_candidate",
            "video_time_s": round(prev["end_frame"] / p.fps, 3),
            "receiver_time_s": round(nxt["start_frame"] / p.fps, 3),
            "initiator_track_id": prev["track_id"],
            "receiver_track_id": nxt["track_id"],
            "gap_seconds": round(gap, 3),
            "review_status": "unreviewed",
            "rule_version": RULE_VERSION,
            "initiator_support_frames": len(prev["frames"]),
            "receiver_support_frames": len(nxt["frames"]),
            "min_distance_over_height": round(min(prev["dists"] + nxt["dists"]), 3),
            "video_id": video_id, "run_id": run_id,
        })
    return (contacts, pd.DataFrame(events, columns=EVENT_CANDIDATE_COLUMNS),
            pd.DataFrame(suppressed, columns=SUPPRESSED_COLUMNS))


def generate(tracks, balls, match_id, fps=25.0, **kwargs):
    """Kit-compatible API: returns (contacts, events)."""
    contacts, events, _ = generate_with_audit(tracks, balls, match_id, CandidateParams(fps=fps, **kwargs))
    return contacts, events


def params_dict(p: CandidateParams) -> dict:
    return asdict(p) | {"rule_version": RULE_VERSION}
