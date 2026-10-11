"""Step 12: file-backed review store with an append-only audit trail.

Layout (per match, under data/annotations/<match_id>/):
  reviewed_events.csv   current reviewed state (one row per event; predictions preserved in predicted_* columns)
  review_audit.jsonl    append-only history: who changed what, before/after, when

Guarantees:
  * original predictions (predicted_*) are never overwritten
  * every save is validated against the taxonomy
  * optimistic concurrency: a save must quote the row's current `version`; stale saves are refused
  * writes are atomic (temp file + rename) under an exclusive file lock
  * re-running candidate generation merges new candidates without touching existing reviews;
    reviews whose candidate disappeared are kept and flagged orphaned
  * reviewers can add events the model missed (source = reviewer_added), needed for recall
This is the local-pilot store. Production uses the PostgreSQL schema (soccer_vision.db).
"""
from __future__ import annotations

import fcntl
import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..events.taxonomy import Taxonomy, load_taxonomy
from ..schemas import REVIEW_COLUMNS, UNKNOWN_PLAYER

EDITABLE = ["event_type", "outcome", "video_time_s", "match_clock_s", "actor_track_id", "actor_player_id",
            "receiver_track_id", "receiver_player_id", "review_status", "review_notes"]


class ConflictError(RuntimeError):
    pass


class ReviewValidationError(ValueError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


class ReviewStore:
    def __init__(self, annotations_dir: Path, match_id: str, taxonomy: Taxonomy | None = None):
        self.dir = Path(annotations_dir)
        self.match_id = match_id
        self.tax = taxonomy or load_taxonomy()
        self.path = self.dir / "reviewed_events.csv"
        self.audit_path = self.dir / "review_audit.jsonl"
        self.lock_path = self.dir / ".review.lock"
        self.dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _locked(self):
        with open(self.lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=REVIEW_COLUMNS)
        df = pd.read_csv(self.path, dtype=str).fillna("")
        for c in REVIEW_COLUMNS:
            if c not in df:
                df[c] = ""
        df["version"] = pd.to_numeric(df["version"], errors="coerce").fillna(0).astype(int)
        return df[REVIEW_COLUMNS]

    def _write(self, df: pd.DataFrame):
        tmp = self.path.with_suffix(".csv.tmp")
        df[REVIEW_COLUMNS].to_csv(tmp, index=False)
        os.replace(tmp, self.path)

    def _audit(self, record: dict):
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def sync_candidates(self, candidates: pd.DataFrame) -> dict:
        """Merge model candidates into the store. Existing reviews are never modified."""
        with self._locked():
            df = self.load()
            known = set(df["event_id"])
            cand_ids = set(candidates["event_id"].astype(str)) if len(candidates) else set()
            new_rows = []
            for r in candidates.itertuples(index=False):
                eid = str(r.event_id)
                if eid in known:
                    continue
                new_rows.append({
                    "event_id": eid, "source": "model", "match_id": str(r.match_id), "event_type": r.event_type,
                    "outcome": "unknown", "video_time_s": r.video_time_s, "match_clock_s": "",
                    "actor_track_id": _s(getattr(r, "initiator_track_id", "")), "actor_player_id": "",
                    "receiver_track_id": _s(getattr(r, "receiver_track_id", "")), "receiver_player_id": "",
                    "review_status": "unreviewed", "review_notes": "", "reviewer_id": "", "reviewed_at": "",
                    "version": 0, "predicted_event_type": r.event_type, "predicted_video_time_s": r.video_time_s,
                    "predicted_initiator_track_id": _s(getattr(r, "initiator_track_id", "")),
                    "predicted_receiver_track_id": _s(getattr(r, "receiver_track_id", "")),
                    "rule_version": _s(getattr(r, "rule_version", "")), "orphaned": "",
                })
            if len(df):
                model_rows = df["source"] == "model"
                df.loc[model_rows, "orphaned"] = df.loc[model_rows, "event_id"].map(
                    lambda e: "" if e in cand_ids else "true")
            if new_rows:
                new = pd.DataFrame(new_rows, columns=REVIEW_COLUMNS)
                for c in REVIEW_COLUMNS:
                    new[c] = new[c].map(_s) if c != "version" else 0
                df = pd.concat([df, new], ignore_index=True)
            df = df.sort_values("video_time_s", key=lambda s: pd.to_numeric(s, errors="coerce"), kind="stable")
            self._write(df)
            orphaned = int((df["orphaned"] == "true").sum())
        return {"added": len(new_rows), "total": len(df), "orphaned": orphaned}

    def _validate(self, row: dict):
        errs = self.tax.validate_review(row["event_type"], row["outcome"], row["review_status"])
        if row["review_status"] in self.tax.confirmed_statuses and self.tax.actor_required(row["event_type"]) \
                and not str(row["actor_player_id"]).strip():
            errs.append(f"confirmed {row['event_type']} needs actor_player_id (or {UNKNOWN_PLAYER})")
        try:
            t = float(row["video_time_s"])
            if t < 0:
                errs.append("video_time_s negative")
        except (TypeError, ValueError):
            errs.append("video_time_s must be a number")
        if errs:
            raise ReviewValidationError("; ".join(errs))

    def save_review(self, event_id: str, changes: dict, reviewer_id: str, expected_version: int) -> dict:
        if not reviewer_id or not reviewer_id.strip():
            raise ReviewValidationError("reviewer_id is required")
        unknown = set(changes).difference(EDITABLE)
        if unknown:
            raise ReviewValidationError(f"not editable: {sorted(unknown)}")
        with self._locked():
            df = self.load()
            hit = df.index[df["event_id"] == event_id]
            if not len(hit):
                raise KeyError(event_id)
            i = hit[0]
            current = int(df.at[i, "version"])
            if current != int(expected_version):
                raise ConflictError(f"{event_id} changed since you loaded it (version {current}, you had "
                                    f"{expected_version}); reload and re-apply")
            before = {k: _s(df.at[i, k]) for k in EDITABLE}
            row = dict(df.loc[i])
            row.update({k: _s(v) for k, v in changes.items()})
            self._validate(row)
            for k, v in changes.items():
                df.at[i, k] = _s(v)
            if df.at[i, "review_status"] == "verified" and _differs_from_prediction(df.loc[i]):
                df.at[i, "review_status"] = "corrected"
            ts = _now()
            df.at[i, "reviewer_id"], df.at[i, "reviewed_at"] = reviewer_id.strip(), ts
            df.at[i, "version"] = current + 1
            after = {k: _s(df.at[i, k]) for k in EDITABLE}
            self._write(df)
            self._audit({"action": "review", "event_id": event_id, "match_id": self.match_id,
                         "reviewer_id": reviewer_id.strip(), "reviewed_at": ts, "version": current + 1,
                         "previous": before, "current": after})
        return after

    def add_missed_event(self, fields: dict, reviewer_id: str) -> str:
        if not reviewer_id or not reviewer_id.strip():
            raise ReviewValidationError("reviewer_id is required")
        eid = f"{self.match_id}-manual-{uuid.uuid4().hex[:10]}"
        row = {c: "" for c in REVIEW_COLUMNS}
        row.update({"event_id": eid, "source": "reviewer_added", "match_id": self.match_id,
                    "outcome": "unknown", "review_status": "verified", "version": "1"})
        row.update({k: _s(v) for k, v in fields.items() if k in EDITABLE})
        self._validate(row)
        ts = _now()
        row["reviewer_id"], row["reviewed_at"] = reviewer_id.strip(), ts
        with self._locked():
            df = self.load()
            new = pd.DataFrame([row], columns=REVIEW_COLUMNS)
            new["version"] = 1
            df = pd.concat([df, new], ignore_index=True)
            df = df.sort_values("video_time_s", key=lambda s: pd.to_numeric(s, errors="coerce"), kind="stable")
            self._write(df)
            self._audit({"action": "add_missed_event", "event_id": eid, "match_id": self.match_id,
                         "reviewer_id": reviewer_id.strip(), "reviewed_at": ts, "previous": None,
                         "current": {k: _s(row[k]) for k in EDITABLE}})
        return eid

    def audit_history(self, event_id: str | None = None) -> list[dict]:
        if not self.audit_path.exists():
            return []
        recs = [json.loads(line) for line in self.audit_path.read_text().splitlines() if line.strip()]
        return [r for r in recs if event_id is None or r["event_id"] == event_id]


def _s(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _differs_from_prediction(r) -> bool:
    if r["source"] != "model":
        return False
    pt = r["predicted_video_time_s"]
    try:
        time_moved = abs(float(r["video_time_s"]) - float(pt)) > 1e-6
    except (TypeError, ValueError):
        time_moved = False
    return (time_moved or _s(r["actor_track_id"]) != _s(r["predicted_initiator_track_id"])
            or _s(r["receiver_track_id"]) != _s(r["predicted_receiver_track_id"]))
