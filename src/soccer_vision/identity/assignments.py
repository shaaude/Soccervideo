"""Step 8 / Step 25: tracklet -> roster identity assignments as a reviewable artifact.

Four keys stay separate: frame detection; tracklet (video_id, run_id, segment_id, track_id);
match appearance; permanent roster player_id. This module never assigns identities on its own:
it builds a review queue (sharpest crops per tracklet) and validates human/OCR assignments
against hard constraints. Jersey-number OCR and re-ID embeddings are pluggable evidence
sources that are NOT implemented here (they need a fine-tuned model and labelled crops).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .team import read_frames

ASSIGNMENT_COLUMNS = ["video_id", "run_id", "segment_id", "track_id", "player_id", "team_id",
                      "jersey_evidence", "identity_confidence", "review_status", "reviewer_id", "reviewed_at", "notes"]
ROSTER_COLUMNS = ["player_id", "team_id", "jersey_number", "start_date", "end_date"]
ID_STATUSES = {"unreviewed", "verified", "corrected", "unknown", "rejected"}


def sharpest_crops(video, tracks: pd.DataFrame, out_dir: Path, per_track: int = 5, candidates: int = 25) -> pd.DataFrame:
    """Save the N sharpest crops per tracklet (variance of Laplacian) for the identity review queue."""
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    plan: dict[int, list] = {}
    for tid, g in tracks.groupby("track_id"):
        g = g.sort_values("frame_index")
        idx = np.unique(np.linspace(0, len(g) - 1, min(candidates, len(g))).round().astype(int))
        for r in g.iloc[idx].itertuples(index=False):
            plan.setdefault(int(r.frame_index), []).append((int(tid), r))
    scored: dict[int, list] = {}
    for f, frame in read_frames(video, set(plan)):
        H, W = frame.shape[:2]
        for tid, r in plan[f]:
            x1, y1, x2, y2 = max(0, int(r.x1)), max(0, int(r.y1)), min(W, int(r.x2)), min(H, int(r.y2))
            if x2 - x1 < 4 or y2 - y1 < 8:
                continue
            crop = frame[y1:y2, x1:x2]
            sharp = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            scored.setdefault(tid, []).append((sharp * (y2 - y1), f, crop))
    rows = []
    for tid, items in scored.items():
        for rank, (score, f, crop) in enumerate(sorted(items, key=lambda x: -x[0])[:per_track], start=1):
            p = out_dir / f"track{tid:05d}_r{rank}_f{f:07d}.jpg"
            cv2.imwrite(str(p), crop)
            rows.append({"track_id": tid, "rank": rank, "frame_index": f, "sharpness": round(score, 1), "crop_path": str(p)})
    return pd.DataFrame(rows, columns=["track_id", "rank", "frame_index", "sharpness", "crop_path"])


def validate_assignments(assign: pd.DataFrame, tracks: pd.DataFrame, roster: pd.DataFrame,
                         match_date: str | None, team_ids: set[str] | None = None,
                         overlap_tolerance_frames: int = 0) -> list[str]:
    """Hard constraints (Step 8.6). Returns a list of violations; empty means consistent."""
    errs = []
    missing = set(ASSIGNMENT_COLUMNS[:9]).difference(assign.columns)
    if missing:
        return [f"assignments missing columns {sorted(missing)}"]
    a = assign.copy()
    a["player_id"] = a["player_id"].fillna("").astype(str)
    bad_status = set(a["review_status"]).difference(ID_STATUSES)
    if bad_status:
        errs.append(f"unknown identity review_status {sorted(bad_status)}")
    if a.duplicated(["video_id", "run_id", "segment_id", "track_id"]).any():
        errs.append("a tracklet has more than one assignment row")
    named = a[(a["player_id"] != "") & a["review_status"].isin(["verified", "corrected", "unreviewed"])]
    r = roster.astype({"player_id": str, "team_id": str})
    if match_date:
        d = pd.Timestamp(match_date)
        active = r[(pd.to_datetime(r["start_date"]) <= d) &
                   (r["end_date"].isna() | (pd.to_datetime(r["end_date"], errors="coerce") >= d))]
    else:
        active = r
        errs.append("match_date unknown: roster validity window cannot be checked")
    if team_ids:
        active = active[active["team_id"].isin(team_ids)]
    for row in named.itertuples(index=False):
        m = active[active["player_id"] == row.player_id]
        if m.empty:
            errs.append(f"track {row.track_id}: player {row.player_id} not on an eligible roster for this match")
            continue
        if str(row.team_id) and str(row.team_id) != "nan" and str(row.team_id) not in set(m["team_id"]):
            errs.append(f"track {row.track_id}: player {row.player_id} not rostered for team {row.team_id}")
        je = str(row.jersey_evidence) if pd.notna(row.jersey_evidence) else ""
        if je and je not in set(m["jersey_number"].astype(str)):
            errs.append(f"track {row.track_id}: jersey evidence {je} contradicts roster number(s) "
                        f"{sorted(set(m['jersey_number'].astype(str)))} for {row.player_id}")
    # No player in two places at once: tracklets assigned to the same player must not overlap in time.
    spans = tracks.groupby("track_id")["frame_index"].agg(["min", "max"])
    for pid, g in named.groupby("player_id"):
        tids = [t for t in g["track_id"].astype(int) if t in spans.index]
        iv = sorted((int(spans.loc[t, "min"]), int(spans.loc[t, "max"]), t) for t in tids)
        for (s0, e0, t0), (s1, e1, t1) in zip(iv, iv[1:]):
            if s1 <= e0 - overlap_tolerance_frames:
                errs.append(f"player {pid}: tracks {t0} and {t1} overlap in frames {s1}-{min(e0, e1)} "
                            f"(same person cannot be in two places)")
    return errs


def resolved_identities(assign: pd.DataFrame) -> pd.DataFrame:
    """Only human-confirmed identities are usable downstream; everything else stays unknown."""
    ok = assign[assign["review_status"].isin(["verified", "corrected"]) & assign["player_id"].notna()
                & (assign["player_id"].astype(str).str.strip() != "")]
    return ok[["video_id", "run_id", "segment_id", "track_id", "player_id", "team_id"]]
