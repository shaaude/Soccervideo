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
