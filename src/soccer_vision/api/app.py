"""Step 16: FastAPI service.

    SOCCER_DATABASE_URL=postgresql+psycopg://... SOCCER_API_KEYS="<key>:admin:alice,<key2>:reviewer:bob" \
    SOCCER_MEDIA_SECRET=<random> uvicorn soccer_vision.api.app:app --host 127.0.0.1

Authorization is enforced here at the backend (Step 17/29), never only in a client. API keys are a
pilot mechanism; replace with your identity provider (OIDC) before any exposure beyond localhost.
Roles: admin, coach, reviewer, analyst, guardian. Guardian access is denied until a verified
guardian-to-player link table exists.

Processing jobs run in a background executor. The default local executor is a thread in this
process (pilot). Production: a Redis+RQ/Celery GPU worker separate from the web server.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..analytics.profiles import build_profiles
from ..events.taxonomy import load_taxonomy
from ..provenance import sha256_file, video_summary
from ..schemas import UNKNOWN_PLAYER
from ..db.models import (AccessAudit, Event, EventReview, Job, Match, MatchVideo, PlayerAppearance, StageRun,
                         Team, TeamRatingSnapshot)
from ..db.session import make_engine, make_sessionmaker

ROLES = {"admin", "coach", "reviewer", "analyst", "guardian"}


class Principal(BaseModel):
    user: str
    role: str


def _keys() -> dict[str, Principal]:
    out = {}
    for item in filter(None, os.environ.get("SOCCER_API_KEYS", "").split(",")):
        key, role, user = item.split(":")
        if role not in ROLES:
            raise RuntimeError(f"Unknown role {role}")
        out[key] = Principal(user=user, role=role)
    return out


# ---------------------------------------------------------------- schemas
class MatchIn(BaseModel):
    match_id: str = Field(min_length=1, max_length=64)
    competition: str | None = None
    division: str | None = None
    season: str | None = None
    age_group: str | None = None
    play_format: str | None = Field(default=None, pattern="^(4v4|7v7|9v9|11v11)$")
    match_date: date | None = None
    home_team_id: str | None = None
    away_team_id: str | None = None
    field_length_m: float | None = Field(default=None, gt=0)
    field_width_m: float | None = Field(default=None, gt=0)
    metadata_verified: bool = False


class VideoIn(BaseModel):
    video_id: str
    file_path: str
    camera_id: str | None = None
    camera_type: str | None = None
    match_clock_offset_seconds: float | None = None
    video_use_permission: str = Field(pattern="^(pending|granted|denied)$")


class ProcessIn(BaseModel):
    video_id: str
    config: str = "config/pipeline.yaml"
    manifest: str | None = None


class EventPatch(BaseModel):
    expected_version: int
    event_type: str | None = None
    outcome: str | None = None
    video_time_seconds: float | None = Field(default=None, ge=0)
    match_clock_seconds: float | None = None
    actor_player_id: str | None = None
    receiver_player_id: str | None = None
    review_status: str | None = None
    notes: str | None = None


class EventNew(BaseModel):
    video_id: str
    event_type: str
    outcome: str = "unknown"
    video_time_seconds: float = Field(ge=0)
    actor_player_id: str | None = None
    receiver_player_id: str | None = None


def _event_dict(e: Event) -> dict:
    return {c.name: getattr(e, c.name) for c in Event.__table__.columns}


def create_app(engine=None, repo: Path | None = None, executor: Callable | None = None) -> FastAPI:
    engine = engine or make_engine()
    SessionLocal = make_sessionmaker(engine)
    repo = Path(repo or os.environ.get("SOCCER_VISION_ROOT", ".")).resolve()
    media_secret = os.environ.get("SOCCER_MEDIA_SECRET", "")
    tax = load_taxonomy()
    app = FastAPI(title="Soccer Vision API", version="0.1.0")
    keys = _keys()

    def db():
        s = SessionLocal()
        try:
            yield s
        finally:
            s.close()

    def principal(request: Request, x_api_key: str = Header(default="")) -> Principal:
        p = keys.get(x_api_key)
        if p is None:
            raise HTTPException(401, "invalid or missing API key")
        with SessionLocal() as s, s.begin():
            s.add(AccessAudit(actor=p.user, role=p.role, action=request.method, resource=str(request.url.path)[:200]))
        return p

    def require(*roles):
        def dep(p: Principal = Depends(principal)):
            if p.role not in roles:
                raise HTTPException(403, f"role '{p.role}' not permitted")
            return p
        return dep

    readers = ("admin", "coach", "reviewer", "analyst")

    # ------------------------------------------------------------ matches/videos
    @app.post("/api/v1/matches", status_code=201)
    def create_match(body: MatchIn, s: Session = Depends(db), p=Depends(require("admin"))):
        if s.get(Match, body.match_id):
            raise HTTPException(409, "match exists")
        if body.home_team_id and body.home_team_id == body.away_team_id:
            raise HTTPException(422, "home and away teams must differ")
        for t in (body.home_team_id, body.away_team_id):
            if t and not s.get(Team, t):
                raise HTTPException(422, f"unknown team {t}")
        s.add(Match(**body.model_dump()))
        s.commit()
        return {"match_id": body.match_id}

    @app.post("/api/v1/matches/{match_id}/videos", status_code=201)
    def register_video(match_id: str, body: VideoIn, s: Session = Depends(db), p=Depends(require("admin"))):
        if not s.get(Match, match_id):
            raise HTTPException(404, "match not found")
        if body.video_use_permission == "denied":
            raise HTTPException(422, "video use permission denied: not registering")
        path = (repo / body.file_path).resolve()
        if repo not in path.parents or not path.exists():
            raise HTTPException(422, "file_path must exist inside the project data area")
        if s.get(MatchVideo, body.video_id):
            raise HTTPException(409, "video_id already registered")
        digest = sha256_file(path)
        if s.scalar(select(MatchVideo).where(MatchVideo.match_id == match_id, MatchVideo.sha256 == digest)):
            raise HTTPException(409, "this file is already registered for the match")
        info = video_summary(path)
        if not info["constant_frame_rate"]:
            raise HTTPException(422, "video is not constant frame rate; normalize first")
        s.add(MatchVideo(video_id=body.video_id, match_id=match_id, file_path=body.file_path, sha256=digest,
                         camera_id=body.camera_id, camera_type=body.camera_type, fps=info["avg_frame_rate"],
                         duration_seconds=info["duration_seconds"],
                         match_clock_offset_seconds=body.match_clock_offset_seconds,
                         video_use_permission=body.video_use_permission))
        s.commit()
        return {"video_id": body.video_id, "fps": info["avg_frame_rate"], "duration_seconds": info["duration_seconds"]}

    # ------------------------------------------------------------ jobs
    def _run_job(job_id: str, match_id: str, video_path: str, cfg: str, manifest: str | None):
        from ..db.importer import import_run
        from ..pipeline.runner import process

        def on_stage(stage, state):
            with SessionLocal() as s2, s2.begin():
                j = s2.get(Job, job_id)
                j.state = state
                j.stages = {**(j.stages or {}), stage: "RUNNING"}
        try:
            st = process(repo, match_id, video_path, repo / cfg, manifest and repo / manifest, on_stage=on_stage)
            import_run(engine, repo, repo / "data" / "processed" / match_id / st["run_id"])
            with SessionLocal() as s2, s2.begin():
                j = s2.get(Job, job_id)
                j.state, j.run_id = "completed", st["run_id"]
                j.stages = {k: v["status"] for k, v in st["stages"].items()}
        except Exception as e:  # recorded, never swallowed silently
            with SessionLocal() as s2, s2.begin():
                j = s2.get(Job, job_id)
                j.state, j.error = "failed", f"{type(e).__name__}: {e}"

    run_executor = executor or (lambda fn, *a: threading.Thread(target=fn, args=a, daemon=True).start())

    @app.post("/api/v1/matches/{match_id}/process", status_code=202)
    def queue_process(match_id: str, body: ProcessIn, s: Session = Depends(db), p=Depends(require("admin"))):
        v = s.get(MatchVideo, body.video_id)
        if not v or v.match_id != match_id:
            raise HTTPException(404, "video not registered for this match")
        if v.video_use_permission != "granted":
            raise HTTPException(422, "video_use_permission must be 'granted' before processing")
        active = s.scalar(select(Job).where(Job.video_id == v.video_id,
                                            Job.state.notin_(["completed", "failed"])))
        if active:
            return {"job_id": active.job_id, "state": active.state, "deduplicated": True}
        job = Job(job_id=uuid.uuid4().hex, match_id=match_id, video_id=v.video_id, state="queued",
                  requested_by=p.user, stages={})
        s.add(job)
        s.commit()
        run_executor(_run_job, job.job_id, match_id, v.file_path, body.config, body.manifest)
        return {"job_id": job.job_id, "state": "queued"}

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str, s: Session = Depends(db), p=Depends(require("admin", "reviewer"))):
        j = s.get(Job, job_id)
        if not j:
            raise HTTPException(404, "job not found")
        stages = []
        if j.run_id:
            stages = [{"stage": r.stage, "status": r.status, "elapsed_seconds": r.elapsed_seconds,
                       "records": r.records, "failure_reason": r.failure_reason, "outputs": r.outputs}
                      for r in s.scalars(select(StageRun).where(StageRun.run_id == j.run_id))]
        return {"job_id": j.job_id, "state": j.state, "run_id": j.run_id, "error": j.error,
                "stage_status": j.stages, "provenance": stages}

    # ------------------------------------------------------------ events
    @app.get("/api/v1/matches/{match_id}/events")
    def list_events(match_id: str, review_status: list[str] = Query(default=[]), event_type: list[str] = Query(default=[]),
                    s: Session = Depends(db), p=Depends(require(*readers))):
        q = select(Event).where(Event.match_id == match_id)
        if p.role in ("coach", "analyst"):  # only reviewers/admins see unconfirmed hypotheses
            q = q.where(Event.review_status.in_(sorted(tax.confirmed_statuses)))
        if review_status:
            q = q.where(Event.review_status.in_(review_status))
        if event_type:
            q = q.where(Event.event_type.in_(event_type))
        return [_event_dict(e) for e in s.scalars(q.order_by(Event.video_time_seconds))]

    def _appearance(s: Session, match_id: str, player_id: str | None):
        if player_id is None or player_id == "":
            return None, False
        if player_id.upper() == UNKNOWN_PLAYER:
            return None, True
        a = s.scalar(select(PlayerAppearance).where(PlayerAppearance.match_id == match_id,
                                                    PlayerAppearance.player_id == player_id))
        if a is None:
            raise HTTPException(422, f"player {player_id} has no appearance in match {match_id}")
        return a.appearance_id, False

    def _check(e: Event):
        errs = tax.validate_review(e.event_type, e.outcome, e.review_status)
        if e.review_status in tax.confirmed_statuses and tax.actor_required(e.event_type) \
                and e.actor_appearance_id is None and not e.actor_unknown:
            errs.append(f"confirmed {e.event_type} needs an actor (or {UNKNOWN_PLAYER})")
        if errs:
            raise HTTPException(422, "; ".join(errs))

    @app.patch("/api/v1/events/{event_id}")
    def patch_event(event_id: str, body: EventPatch, s: Session = Depends(db), p=Depends(require("admin", "reviewer"))):
        e = s.get(Event, event_id, with_for_update=True)
        if not e:
            raise HTTPException(404, "event not found")
        if e.version != body.expected_version:
            raise HTTPException(409, f"event changed (version {e.version}); reload")
        before = {k: v for k, v in _event_dict(e).items() if k != "original_prediction"}
        data = body.model_dump(exclude_unset=True, exclude={"expected_version", "notes"})
        for k in ("event_type", "outcome", "video_time_seconds", "match_clock_seconds", "review_status"):
            if k in data:
                setattr(e, k, data[k])
        if "actor_player_id" in data:
            e.actor_appearance_id, e.actor_unknown = _appearance(s, e.match_id, data["actor_player_id"])
        if "receiver_player_id" in data:
            e.receiver_appearance_id, _ = _appearance(s, e.match_id, data["receiver_player_id"])
        _check(e)
        e.version += 1
        after = {k: v for k, v in _event_dict(e).items() if k != "original_prediction"}
        changed = {k: after[k] for k in after if after[k] != before[k]}
        s.add(EventReview(event_id=event_id, reviewer_id=p.user, previous_values={k: before[k] for k in changed},
                          corrected_values={**changed, **({"notes": body.notes} if body.notes else {})}))
        s.commit()
        return {k: (str(v) if isinstance(v, date) else v) for k, v in after.items()}

    @app.post("/api/v1/matches/{match_id}/events", status_code=201)
    def add_event(match_id: str, body: EventNew, s: Session = Depends(db), p=Depends(require("admin", "reviewer"))):
        v = s.get(MatchVideo, body.video_id)
        if not v or v.match_id != match_id:
            raise HTTPException(404, "video not registered for this match")
        e = Event(event_id=f"{match_id}-manual-{uuid.uuid4().hex[:10]}", match_id=match_id, video_id=v.video_id,
                  source="reviewer_added", video_time_seconds=body.video_time_seconds, event_type=body.event_type,
                  outcome=body.outcome, review_status="verified", version=1, original_prediction={})
        e.actor_appearance_id, e.actor_unknown = _appearance(s, match_id, body.actor_player_id)
        e.receiver_appearance_id, _ = _appearance(s, match_id, body.receiver_player_id)
        _check(e)
        s.add(e)
        s.flush()
        s.add(EventReview(event_id=e.event_id, reviewer_id=p.user, previous_values=None,
                          corrected_values={"action": "add_missed_event", **body.model_dump()}))
        s.commit()
        return {"event_id": e.event_id}

    # ------------------------------------------------------------ players / teams
    def _player_events(s: Session, player_id: str) -> pd.DataFrame:
        q = (select(Event, PlayerAppearance).join(PlayerAppearance, Event.actor_appearance_id == PlayerAppearance.appearance_id)
             .where(PlayerAppearance.player_id == player_id))
        rows = [{"event_id": e.event_id, "match_id": e.match_id, "event_type": e.event_type, "outcome": e.outcome,
                 "actor_player_id": a.player_id, "review_status": e.review_status,
                 "video_id": e.video_id, "video_time_s": e.video_time_seconds} for e, a in s.execute(q)]
        return pd.DataFrame(rows, columns=["event_id", "match_id", "event_type", "outcome", "actor_player_id",
                                           "review_status", "video_id", "video_time_s"])

    @app.get("/api/v1/players/{player_id}/profile")
    def player_profile(player_id: str, s: Session = Depends(db), p=Depends(require(*readers))):
        ev = _player_events(s, player_id)
        apps = s.scalars(select(PlayerAppearance).where(PlayerAppearance.player_id == player_id)).all()
        exposure = pd.DataFrame([{"match_id": a.match_id, "player_id": a.player_id,
                                  "reviewed_minutes": a.reviewed_minutes_observed} for a in apps
                                 if a.reviewed_minutes_observed is not None and a.identity_reviewed])
        profiles, report = build_profiles(ev, tax, exposure if len(exposure) else None)
        prof = profiles.iloc[0].to_dict() if len(profiles) else None
        return {"player_id": player_id, "verified_only": True, "profile": _clean(prof), "coverage": report,
                "appearances": len(apps)}

    def sign(video_id: str, start: float, end: float, expires: int) -> str:
        msg = f"{video_id}|{start:.3f}|{end:.3f}|{expires}".encode()
        return hmac.new(media_secret.encode(), msg, hashlib.sha256).hexdigest()

    @app.get("/api/v1/players/{player_id}/clips")
    def player_clips(player_id: str, ttl: int = Query(default=300, ge=30, le=3600), s: Session = Depends(db),
                     p=Depends(require(*readers))):
        if not media_secret:
            raise HTTPException(503, "SOCCER_MEDIA_SECRET not configured; media access disabled")
        ev = _player_events(s, player_id)
        ev = ev[ev["review_status"].isin(tax.confirmed_statuses)]
        exp = int(time.time()) + ttl
        out = []
        for r in ev.itertuples(index=False):
            st, en = max(0.0, r.video_time_s - 4), r.video_time_s + 6
            sig = sign(r.video_id, st, en, exp)
            out.append({"event_id": r.event_id, "event_type": r.event_type, "video_id": r.video_id,
                        "start_seconds": st, "end_seconds": en,
                        "url": f"/api/v1/media/{r.video_id}?start={st:.3f}&end={en:.3f}&expires={exp}&sig={sig}"})
        return out

    @app.get("/api/v1/media/{video_id}")
    def media(video_id: str, start: float, end: float, expires: int, sig: str, s: Session = Depends(db)):
        if not media_secret or expires < time.time() or \
                not hmac.compare_digest(sig, sign(video_id, start, end, expires)):
            raise HTTPException(403, "invalid or expired media link")
        v = s.get(MatchVideo, video_id)
        if not v:
            raise HTTPException(404, "video not found")
        return FileResponse(repo / v.file_path, media_type="video/mp4",
                            headers={"X-Clip-Start": f"{start:.3f}", "X-Clip-End": f"{end:.3f}",
                                     "Cache-Control": "private, no-store"})

    @app.get("/api/v1/teams/{team_id}/analysis")
    def team_analysis(team_id: str, rating_system: str = "elo_v1", s: Session = Depends(db),
                      p=Depends(require(*readers))):
        if not s.get(Team, team_id):
            raise HTTPException(404, "team not found")
        matches = s.scalars(select(Match).where((Match.home_team_id == team_id) | (Match.away_team_id == team_id))
                            .order_by(Match.match_date)).all()

        def pre_match(tid, d):
            if d is None:
                return None
            snap = s.scalar(select(TeamRatingSnapshot).where(
                TeamRatingSnapshot.team_id == tid, TeamRatingSnapshot.rating_system == rating_system,
                TeamRatingSnapshot.effective_at < pd.Timestamp(d, tz="UTC").to_pydatetime())
                .order_by(TeamRatingSnapshot.effective_at.desc()))
            return None if snap is None else {"rating": snap.rating, "effective_at": snap.effective_at.isoformat(),
                                              "input_cutoff": snap.input_cutoff.isoformat()}
        out = []
        for m in matches:
            opp = m.away_team_id if m.home_team_id == team_id else m.home_team_id
            counts = {}
            for e, a in s.execute(select(Event, PlayerAppearance)
                                  .join(PlayerAppearance, Event.actor_appearance_id == PlayerAppearance.appearance_id)
                                  .where(Event.match_id == m.match_id, PlayerAppearance.team_id == team_id,
                                         Event.review_status.in_(sorted(tax.confirmed_statuses)))):
                counts[e.event_type] = counts.get(e.event_type, 0) + 1
            gf, ga = (m.home_score, m.away_score) if m.home_team_id == team_id else (m.away_score, m.home_score)
            out.append({"match_id": m.match_id, "match_date": str(m.match_date) if m.match_date else None,
                        "opponent_team_id": opp, "goals_for": gf, "goals_against": ga,
                        "team_pre_match_rating": pre_match(team_id, m.match_date),
                        "opponent_pre_match_rating": pre_match(opp, m.match_date),
                        "verified_event_counts": counts, "play_format": m.play_format})
        return {"team_id": team_id, "rating_system": rating_system, "matches": out,
                "note": "Event counts are reviewed events only; ratings are strictly pre-match snapshots."}

    return app


def _clean(d):
    if d is None:
        return None
    return {k: (None if (isinstance(v, float) and v != v) else (v.item() if hasattr(v, "item") else v)) for k, v in d.items()}


def __getattr__(name):  # `uvicorn soccer_vision.api.app:app` builds the app lazily from the environment
    if name == "app":
        return create_app()
    raise AttributeError(name)
