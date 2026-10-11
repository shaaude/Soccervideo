"""Step 24: one orchestrated, resumable, provenance-recording processing command.

    soccer-vision process --match-id match001 --video data/normalized/match001.mp4 --config config/pipeline.yaml

Run directory: data/processed/<match_id>/<run_hash>/  where run_hash = sha256(video sha256, weight
checksums, resolved config, code revision)[:16]. Each stage records status (PENDING, RUNNING,
SUCCEEDED, FAILED, SKIPPED), input/output checksums, elapsed time, record counts and failure reason in
manifest.json. A completed stage is reused only when its recorded inputs and outputs still match.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from ..manifest import load_manifest
from ..provenance import canonical_json, code_revision, sha256_file, sha256_text

STAGES = ["validate", "detect_track", "calibrate", "identify", "candidates", "temporal_model",
          "highlights", "review_queue", "aggregate", "report"]
# Job states exposed by the API (Step 16) for each stage.
JOB_STATE_FOR_STAGE = {"validate": "ingesting", "detect_track": "detecting", "calibrate": "calibrating",
                       "identify": "identifying", "candidates": "spotting", "temporal_model": "spotting",
                       "highlights": "spotting", "review_queue": "review", "aggregate": "aggregation",
                       "report": "aggregation"}


class StageSkipped(Exception):
    """Raised by a stage to record SKIPPED with an explicit reason (missing != zero)."""


@dataclass
class Context:
    repo: Path
    match_id: str
    video: Path
    manifest: dict
    config: dict
    run_dir: Path
    run_id: str
    annotations_dir: Path
    outputs: dict = field(default_factory=dict)  # stage -> {name: path}
    manifest_path: Path | None = None

    def path(self, name: str) -> Path:
        return self.run_dir / name

    def rel(self, p: Path) -> str:
        try:
            return str(Path(p).resolve().relative_to(self.repo.resolve()))
        except ValueError:
            return str(p)


def _abs(repo: Path, p) -> Path | None:
    if p in (None, ""):
        return None
    p = Path(p)
    return p if p.is_absolute() else repo / p


def compute_run(repo: Path, video: Path, config: dict) -> tuple[str, dict]:
    det = config["detection"]
    weights = {}
    for key in ("player_weights", "ball_weights"):
        wp = _abs(repo, det.get(key))
        if wp is not None:
            if not wp.exists():
                raise FileNotFoundError(f"Missing model checkpoint for {key}: {wp}")
            weights[key] = sha256_file(wp)
    cm = _abs(repo, det["class_map"])
    basis = {"video_sha256": sha256_file(video), "weights": weights,
             "class_map_sha256": sha256_file(cm), "config": config, "code_revision": code_revision()}
    return sha256_text(canonical_json(basis))[:16], basis


class Pipeline:
    def __init__(self, repo: Path, match_id: str, video: Path, config_path: Path, manifest_path: Path | None = None,
                 on_stage: Callable[[str, str], None] | None = None):
        self.repo = Path(repo).resolve()
        cfg = yaml.safe_load(Path(config_path).read_text())
        cfg["device"] = os.environ.get("SOCCER_DEVICE", cfg.get("device", "cpu"))
        manifest_path = manifest_path or self.repo / "config" / f"{match_id}.json"
        manifest = load_manifest(manifest_path)
        if manifest["match_id"] != match_id:
            raise ValueError(f"Manifest match_id {manifest['match_id']} != --match-id {match_id}")
        video = Path(video)
        video = video if video.is_absolute() else self.repo / video
        if not video.exists():
            raise FileNotFoundError(video)
        run_id, basis = compute_run(self.repo, video, cfg)
        run_dir = self.repo / "data" / "processed" / match_id / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self.ctx = Context(self.repo, match_id, video, manifest, cfg, run_dir, run_id,
                           self.repo / "data" / "annotations" / match_id, manifest_path=Path(manifest_path))
        self.basis = basis
        self.manifest_path = run_dir / "manifest.json"
        self.on_stage = on_stage
        self.state = self._load_state(manifest_path)

    def _load_state(self, match_manifest_path):
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {"run_id": self.ctx.run_id, "match_id": self.ctx.match_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "video": self.ctx.rel(self.ctx.video), "match_manifest": self.ctx.rel(Path(match_manifest_path)),
                "basis": {k: v for k, v in self.basis.items() if k != "config"},
                "config": self.ctx.config,
                "stages": {s: {"status": "PENDING"} for s in STAGES}}

    def _save(self):
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2, default=str))
        tmp.replace(self.manifest_path)

    def _checksums(self, files: dict) -> dict:
        out = {}
        for k, p in files.items():
            p = Path(p)
            out[k] = {"path": self.ctx.rel(p), "sha256": sha256_file(p) if p.is_file() else None}
        return out

    def _reusable(self, name, inputs) -> bool:
        st = self.state["stages"].get(name, {})
        if st.get("status") != "SUCCEEDED" or st.get("inputs") != inputs:
            return False
        for k, v in st.get("outputs", {}).items():
            p = self.repo / v["path"]
            if v["sha256"] is None or not p.exists() or sha256_file(p) != v["sha256"]:
                return False
        return True

    def run(self, stop_after: str | None = None, force: bool = False) -> dict:
        from . import stages as S
        funcs = {s: getattr(S, f"stage_{s}") for s in STAGES}
        for name in STAGES:
            inputs = self._checksums(S.stage_inputs(name, self.ctx))
            rec = self.state["stages"].setdefault(name, {"status": "PENDING"})
            if not force and self._reusable(name, inputs):
                self.ctx.outputs[name] = {k: self.repo / v["path"] for k, v in rec["outputs"].items()}
                rec["reused_at"] = datetime.now(timezone.utc).isoformat()
                self._save()
                if stop_after == name:
                    break
                continue
            if self.on_stage:
                self.on_stage(name, JOB_STATE_FOR_STAGE[name])
            rec.clear()
            rec.update({"status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(), "inputs": inputs})
            self._save()
            t0 = time.time()
            try:
                result = funcs[name](self.ctx) or {}
                outs = result.get("outputs", {})
                self.ctx.outputs[name] = outs
                rec.update({"status": "SUCCEEDED", "outputs": self._checksums(outs),
                            "records": result.get("records"), "notes": result.get("notes")})
            except StageSkipped as s:
                self.ctx.outputs[name] = {}
                rec.update({"status": "SKIPPED", "outputs": {}, "reason": str(s)})
            except Exception as e:
                rec.update({"status": "FAILED", "failure_reason": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc(limit=8)})
                rec["elapsed_seconds"] = round(time.time() - t0, 3)
                self._save()
                raise
            rec["elapsed_seconds"] = round(time.time() - t0, 3)
            rec["finished_at"] = datetime.now(timezone.utc).isoformat()
            self._save()
            if stop_after == name:
                break
        return self.state


def process(repo, match_id, video, config, manifest=None, stop_after=None, force=False, on_stage=None) -> dict:
    return Pipeline(Path(repo), match_id, Path(video), Path(config), manifest and Path(manifest),
                    on_stage=on_stage).run(stop_after=stop_after, force=force)


def summarize(state: dict) -> pd.DataFrame:
    rows = []
    for s in STAGES:
        r = state["stages"].get(s, {})
        rows.append({"stage": s, "status": r.get("status"), "records": r.get("records"),
                     "elapsed_s": r.get("elapsed_seconds"),
                     "note": r.get("reason") or r.get("failure_reason") or (r.get("notes") or "")})
    return pd.DataFrame(rows)
