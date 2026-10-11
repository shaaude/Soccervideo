"""Step 13 / 15 / 24 / 27: persistent schema (PostgreSQL in production).

Logically distinct stores: observations live in Parquet/CSV per immutable run; this database holds
match metadata, identities, events (predictions and reviewed state), review history, provenance,
coverage, and external-source mappings. Child identifiers are minimized (display_name only).
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (JSON, BigInteger, Boolean, CheckConstraint, Date, DateTime, Float, ForeignKey,
                        Index, Integer, String, Text, UniqueConstraint, func)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

JSONType = JSON().with_variant(JSONB(), "postgresql")

REVIEW_STATUSES = ("unreviewed", "verified", "corrected", "rejected", "unobservable")
JOB_STATES = ("queued", "ingesting", "detecting", "tracking", "calibrating", "identifying", "spotting",
              "review", "aggregation", "completed", "failed")
STAGE_STATES = ("PENDING", "RUNNING", "SUCCEEDED", "FAILED", "SKIPPED")


def _in(col: str, values) -> str:
    return f"{col} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"
    team_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    team_name: Mapped[str] = mapped_column(String(200))
    club_name: Mapped[str | None] = mapped_column(String(200))
    age_group: Mapped[str | None] = mapped_column(String(16))


class Player(Base):
    __tablename__ = "players"
    player_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(120))  # minimize child identifiers: no DOB/address


class RosterMembership(Base):
    __tablename__ = "roster_memberships"
    __table_args__ = (CheckConstraint("end_date IS NULL OR end_date >= start_date", name="ck_roster_dates"),)
    roster_membership_id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    player_id: Mapped[str] = mapped_column(ForeignKey("players.player_id"), index=True)
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.team_id"), index=True)
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)
    jersey_number: Mapped[str | None] = mapped_column(String(8))


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (CheckConstraint("home_team_id IS NULL OR home_team_id <> away_team_id", name="ck_match_teams"),
                      CheckConstraint("field_length_m IS NULL OR field_length_m > 0", name="ck_match_len"),
                      CheckConstraint("field_width_m IS NULL OR field_width_m > 0", name="ck_match_wid"))
    match_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    competition: Mapped[str | None] = mapped_column(String(64))
    division: Mapped[str | None] = mapped_column(String(64))
    season: Mapped[str | None] = mapped_column(String(32))
    age_group: Mapped[str | None] = mapped_column(String(16))
    play_format: Mapped[str | None] = mapped_column(String(8))
    match_date: Mapped[date | None] = mapped_column(Date)
    home_team_id: Mapped[str | None] = mapped_column(ForeignKey("teams.team_id"))
    away_team_id: Mapped[str | None] = mapped_column(ForeignKey("teams.team_id"))
    field_length_m: Mapped[float | None] = mapped_column(Float)
    field_width_m: Mapped[float | None] = mapped_column(Float)
    home_score: Mapped[int | None] = mapped_column(Integer)
    away_score: Mapped[int | None] = mapped_column(Integer)
    metadata_verified: Mapped[bool] = mapped_column(Boolean, default=False)


class MatchVideo(Base):
    __tablename__ = "match_videos"
    __table_args__ = (UniqueConstraint("match_id", "sha256", name="uq_video_sha"),
                      CheckConstraint("fps > 0", name="ck_video_fps"),
                      CheckConstraint(_in("video_use_permission", ("pending", "granted", "denied")), name="ck_video_perm"))
    video_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.match_id"), index=True)
    file_path: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64))
    camera_id: Mapped[str | None] = mapped_column(String(32))
    camera_type: Mapped[str | None] = mapped_column(String(32))
    fps: Mapped[float] = mapped_column(Float)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    match_clock_offset_seconds: Mapped[float | None] = mapped_column(Float)
    video_use_permission: Mapped[str] = mapped_column(String(16), default="pending")


class PlayerAppearance(Base):
    __tablename__ = "player_appearances"
    __table_args__ = (UniqueConstraint("match_id", "player_id", name="uq_appearance"),
                      CheckConstraint("minutes_played IS NULL OR minutes_played >= 0", name="ck_minutes"))
    appearance_id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.match_id"), index=True)
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.team_id"))
    player_id: Mapped[str] = mapped_column(ForeignKey("players.player_id"), index=True)
    minutes_played: Mapped[float | None] = mapped_column(Float)
    reviewed_minutes_observed: Mapped[float | None] = mapped_column(Float)
    identity_reviewed: Mapped[bool] = mapped_column(Boolean, default=False)


class ProcessingRun(Base):
    __tablename__ = "processing_runs"
    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)  # run hash
    video_id: Mapped[str] = mapped_column(ForeignKey("match_videos.video_id"), index=True)
    code_revision: Mapped[str] = mapped_column(String(80))
    config_sha256: Mapped[str] = mapped_column(String(64))
    weights: Mapped[dict] = mapped_column(JSONType, default=dict)
    manifest_path: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class StageRun(Base):
    __tablename__ = "stage_runs"
    __table_args__ = (UniqueConstraint("run_id", "stage", name="uq_stage"),
                      CheckConstraint(_in("status", STAGE_STATES), name="ck_stage_status"))
    stage_run_id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("processing_runs.run_id"), index=True)
    stage: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(12))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    elapsed_seconds: Mapped[float | None] = mapped_column(Float)
    records: Mapped[int | None] = mapped_column(Integer)
    inputs: Mapped[dict] = mapped_column(JSONType, default=dict)
    outputs: Mapped[dict] = mapped_column(JSONType, default=dict)
    failure_reason: Mapped[str | None] = mapped_column(Text)


class CoverageInterval(Base):
    __tablename__ = "coverage_intervals"
    __table_args__ = (CheckConstraint("end_seconds > start_seconds", name="ck_cov_order"),
                      CheckConstraint(_in("coverage_type", ("camera", "annotation", "calibration")), name="ck_cov_type"))
    coverage_id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("match_videos.video_id"), index=True)
    coverage_type: Mapped[str] = mapped_column(String(16))
    start_seconds: Mapped[float] = mapped_column(Float)
    end_seconds: Mapped[float] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)


class TrackletAssignment(Base):
    __tablename__ = "tracklet_assignments"
    __table_args__ = (CheckConstraint(_in("review_status", ("unreviewed", "verified", "corrected", "unknown", "rejected")),
                                      name="ck_tracklet_status"),
                      CheckConstraint("identity_confidence IS NULL OR (identity_confidence >= 0 AND identity_confidence <= 1)",
                                      name="ck_tracklet_conf"))
    video_id: Mapped[str] = mapped_column(ForeignKey("match_videos.video_id"), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("processing_runs.run_id"), primary_key=True)
    track_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    segment_id: Mapped[int] = mapped_column(Integer, primary_key=True, default=0)
    appearance_id: Mapped[int | None] = mapped_column(ForeignKey("player_appearances.appearance_id"))
    identity_confidence: Mapped[float | None] = mapped_column(Float)
    evidence: Mapped[dict] = mapped_column(JSONType, default=dict)
    review_status: Mapped[str] = mapped_column(String(16), default="unreviewed")


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (
        CheckConstraint(_in("review_status", REVIEW_STATUSES), name="ck_event_status"),
        CheckConstraint("video_time_seconds >= 0", name="ck_event_time"),
        CheckConstraint("event_confidence IS NULL OR (event_confidence >= 0 AND event_confidence <= 1)", name="ck_event_conf"),
        CheckConstraint("actor_confidence IS NULL OR (actor_confidence >= 0 AND actor_confidence <= 1)", name="ck_actor_conf"),
        # A machine-only candidate type can never be marked as confirmed.
        CheckConstraint("NOT (event_type = 'ball_transfer_candidate' AND review_status IN ('verified','corrected'))",
                        name="ck_candidate_not_confirmed"),
        Index("ix_events_match_time", "match_id", "video_time_seconds"),
        Index("ix_events_actor", "actor_appearance_id"),
        Index("ix_events_review", "review_status"),
    )
    event_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.match_id"))
    video_id: Mapped[str] = mapped_column(ForeignKey("match_videos.video_id"))
    run_id: Mapped[str | None] = mapped_column(ForeignKey("processing_runs.run_id"))
    source: Mapped[str] = mapped_column(String(16), default="model")  # model | reviewer_added
    video_time_seconds: Mapped[float] = mapped_column(Float)
    match_clock_seconds: Mapped[float | None] = mapped_column(Float)
    event_type: Mapped[str] = mapped_column(String(40))
    outcome: Mapped[str] = mapped_column(String(24), default="unknown")
    actor_track_id: Mapped[int | None] = mapped_column(Integer)
    receiver_track_id: Mapped[int | None] = mapped_column(Integer)
    actor_appearance_id: Mapped[int | None] = mapped_column(ForeignKey("player_appearances.appearance_id"))
    receiver_appearance_id: Mapped[int | None] = mapped_column(ForeignKey("player_appearances.appearance_id"))
    actor_unknown: Mapped[bool] = mapped_column(Boolean, default=False)
    event_confidence: Mapped[float | None] = mapped_column(Float)
    actor_confidence: Mapped[float | None] = mapped_column(Float)
    review_status: Mapped[str] = mapped_column(String(16), default="unreviewed")
    model_version: Mapped[str | None] = mapped_column(String(64))
    original_prediction: Mapped[dict] = mapped_column(JSONType, default=dict)  # never updated after insert
    version: Mapped[int] = mapped_column(Integer, default=0)  # optimistic concurrency


class EventReview(Base):
    __tablename__ = "event_reviews"
    review_id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("events.event_id"), index=True)
    reviewer_id: Mapped[str] = mapped_column(String(64))
    previous_values: Mapped[dict | None] = mapped_column(JSONType)
    corrected_values: Mapped[dict] = mapped_column(JSONType)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ExternalIdMapping(Base):
    """Step 15: canonical keys across result sources. Never join on team name alone."""
    __tablename__ = "external_id_mappings"
    __table_args__ = (UniqueConstraint("source_system", "source_entity_type", "source_entity_id", name="uq_ext"),
                      CheckConstraint(_in("mapping_status", ("proposed", "verified", "rejected", "collision")),
                                      name="ck_ext_status"),
                      CheckConstraint(_in("source_entity_type", ("team", "match", "player", "competition")),
                                      name="ck_ext_type"))
    mapping_id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    source_system: Mapped[str] = mapped_column(String(40))
    source_entity_type: Mapped[str] = mapped_column(String(16))
    source_entity_id: Mapped[str] = mapped_column(String(128))
    internal_entity_id: Mapped[str | None] = mapped_column(String(96))
    mapping_status: Mapped[str] = mapped_column(String(12), default="proposed")
    match_basis: Mapped[str | None] = mapped_column(String(64))  # provider_id | teams+date+competition+score
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reviewer_id: Mapped[str | None] = mapped_column(String(64))


class TeamRatingSnapshot(Base):
    """Step 15/27: ratings with effective date and input cutoff, so analyses use PRE-match ratings only."""
    __tablename__ = "team_rating_snapshots"
    __table_args__ = (UniqueConstraint("team_id", "rating_system", "effective_at", name="uq_rating"),
                      CheckConstraint("input_cutoff <= effective_at", name="ck_rating_cutoff"))
    snapshot_id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.team_id"), index=True)
    rating_system: Mapped[str] = mapped_column(String(32))  # e.g. elo_v1, dixon_coles_v1
    rating: Mapped[float] = mapped_column(Float)
    effective_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    input_cutoff: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source_system: Mapped[str | None] = mapped_column(String(40))


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (CheckConstraint(_in("state", JOB_STATES), name="ck_job_state"),)
    job_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.match_id"), index=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("match_videos.video_id"))
    state: Mapped[str] = mapped_column(String(16), default="queued")
    run_id: Mapped[str | None] = mapped_column(String(64))
    error: Mapped[str | None] = mapped_column(Text)
    stages: Mapped[dict] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    requested_by: Mapped[str | None] = mapped_column(String(64))


class AccessAudit(Base):
    """Step 29: who accessed or changed what (API layer writes here)."""
    __tablename__ = "access_audit"
    audit_id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(24))
    action: Mapped[str] = mapped_column(String(64))
    resource: Mapped[str] = mapped_column(String(200))
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
