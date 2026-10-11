"""Load a processing run (manifest + candidates) into the database. Idempotent."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..manifest import load_manifest
from .models import Event, Match, MatchVideo, ProcessingRun, StageRun


def _date(v):
    return date.fromisoformat(v) if v else None


def import_run(engine, repo: Path, run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    state = json.loads((run_dir / "manifest.json").read_text())
    m = load_manifest(repo / state["match_manifest"])
    counts = {"events_inserted": 0, "events_existing": 0, "stages": 0}
    with Session(engine) as s, s.begin():
        if s.get(Match, m["match_id"]) is None:
            s.add(Match(match_id=m["match_id"], competition=m.get("competition"), division=m.get("division"),
                        season=m.get("season"), age_group=m.get("age_group"), play_format=m.get("play_format"),
                        match_date=_date(m.get("match_date")), home_team_id=m.get("home_team_id"),
                        away_team_id=m.get("away_team_id"), field_length_m=m.get("field_length_m"),
                        field_width_m=m.get("field_width_m")))
            s.flush()
        vid = m["video_id"]
        if s.get(MatchVideo, vid) is None:
            info = json.loads((run_dir / "video_info.json").read_text())["summary"] if (run_dir / "video_info.json").exists() else {}
            s.add(MatchVideo(video_id=vid, match_id=m["match_id"], file_path=state["video"],
                             sha256=state["basis"]["video_sha256"], camera_id=m.get("camera_id"),
                             camera_type=m.get("camera_type"), fps=float(m["normalized_fps"]),
                             duration_seconds=info.get("duration_seconds"),
                             match_clock_offset_seconds=m.get("match_clock_offset_seconds"),
                             video_use_permission=m.get("video_use_permission", "pending")))
            s.flush()
        run_id = state["run_id"]
        if s.get(ProcessingRun, run_id) is None:
            s.add(ProcessingRun(run_id=run_id, video_id=vid, code_revision=state["basis"]["code_revision"],
                                config_sha256=state["basis"].get("class_map_sha256", ""),
                                weights=state["basis"].get("weights", {}),
                                manifest_path=str(run_dir / "manifest.json")))
            s.flush()
        for name, rec in state["stages"].items():
            row = s.scalar(select(StageRun).where(StageRun.run_id == run_id, StageRun.stage == name))
            if row is None:
                row = StageRun(run_id=run_id, stage=name)
                s.add(row)
            row.status = rec.get("status", "PENDING")
            row.elapsed_seconds = rec.get("elapsed_seconds")
            row.records = rec.get("records")
            row.inputs = rec.get("inputs", {})
            row.outputs = rec.get("outputs", {})
            row.failure_reason = rec.get("failure_reason") or rec.get("reason")
            counts["stages"] += 1
        ev_path = run_dir / "event_candidates.csv"
        if ev_path.exists():
            ev = pd.read_csv(ev_path, dtype={"event_id": str, "match_id": str})
            for r in ev.itertuples(index=False):
                if s.get(Event, r.event_id) is not None:
                    counts["events_existing"] += 1
                    continue
                pred = {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                        for k, v in r._asdict().items()}
                s.add(Event(event_id=r.event_id, match_id=m["match_id"], video_id=vid, run_id=run_id, source="model",
                            video_time_seconds=float(r.video_time_s), event_type=r.event_type, outcome="unknown",
                            actor_track_id=int(r.initiator_track_id), receiver_track_id=int(r.receiver_track_id),
                            review_status="unreviewed", model_version=r.rule_version, original_prediction=pred))
                counts["events_inserted"] += 1
    return counts


def import_reviews(engine, annotations_dir: Path, match_id: str) -> dict:
    """Apply the file-based pilot review store to the database (one-way, idempotent).

    Only rows whose DB version is older than the file version are applied. Roster actors must already
    have a player_appearance in the match; otherwise the row is skipped and reported, never guessed.
    Each applied change writes an event_reviews record carrying the file audit's reviewer.
    """
    from ..review.store import ReviewStore
    from ..schemas import UNKNOWN_PLAYER
    from .models import EventReview, PlayerAppearance

    store = ReviewStore(Path(annotations_dir), match_id)
    df = store.load()
    out = {"applied": 0, "added": 0, "unchanged": 0, "skipped": []}

    def appearance(s, pid):
        if not pid:
            return None, False, None
        if pid.upper() == UNKNOWN_PLAYER:
            return None, True, None
        a = s.scalar(select(PlayerAppearance).where(PlayerAppearance.match_id == match_id,
                                                    PlayerAppearance.player_id == pid))
        return (a.appearance_id, False, None) if a else (None, False, f"no appearance for player {pid}")

    with Session(engine) as s, s.begin():
        for r in df.itertuples(index=False):
            version = int(r.version)
            e = s.get(Event, r.event_id)
            if e is None and r.source == "reviewer_added":
                vid = s.scalar(select(MatchVideo.video_id).where(MatchVideo.match_id == match_id))
                e = Event(event_id=r.event_id, match_id=match_id, video_id=vid, source="reviewer_added",
                          video_time_seconds=float(r.video_time_s), event_type=r.event_type, outcome=r.outcome,
                          review_status="unreviewed", version=0, original_prediction={})
                s.add(e)
                out["added"] += 1
            if e is None:
                out["skipped"].append(f"{r.event_id}: not in database (import its run first)")
                continue
            if e.version >= version:
                out["unchanged"] += 1
                continue
            actor_id, unknown, err = appearance(s, r.actor_player_id)
            recv_id, _, err2 = appearance(s, r.receiver_player_id)
            if err or err2:
                out["skipped"].append(f"{r.event_id}: {err or err2}")
                continue
            before = {"event_type": e.event_type, "outcome": e.outcome, "review_status": e.review_status,
                      "video_time_seconds": e.video_time_seconds}
            e.event_type, e.outcome, e.review_status = r.event_type, r.outcome, r.review_status
            e.video_time_seconds = float(r.video_time_s)
            e.match_clock_seconds = float(r.match_clock_s) if r.match_clock_s else None
            e.actor_appearance_id, e.actor_unknown, e.receiver_appearance_id = actor_id, unknown, recv_id
            e.version = version
            s.add(EventReview(event_id=e.event_id, reviewer_id=r.reviewer_id or "unknown",
                              previous_values=before,
                              corrected_values={"event_type": r.event_type, "outcome": r.outcome,
                                                "review_status": r.review_status, "actor_player_id": r.actor_player_id,
                                                "source": "file_review_store", "notes": r.review_notes}))
            out["applied"] += 1
    return out
