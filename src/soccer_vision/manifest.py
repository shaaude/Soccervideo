"""Match manifest (Step 5): explicit, verified metadata. Nothing is inferred silently."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REQUIRED_KEYS = [
    "match_id", "season", "competition", "age_group", "division", "home_team_id", "away_team_id",
    "match_date", "field_length_m", "field_width_m", "camera_id", "camera_type",
    "normalized_video", "normalized_fps", "match_clock_offset_seconds", "video_use_permission",
]
# Values that must be resolved from a verified source before production processing.
PRODUCTION_REQUIRED = [
    "home_team_id", "away_team_id", "match_date", "field_length_m", "field_width_m",
    "match_clock_offset_seconds",
]
PERMISSION_VALUES = {"pending", "granted", "denied"}
PLAY_FORMATS = {"4v4", "7v7", "9v9", "11v11"}


@dataclass
class ManifestReport:
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def load_manifest(path) -> dict:
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    m.setdefault("video_id", f"{m.get('match_id')}-{m.get('camera_id')}")
    return m


def validate_manifest(m: dict, production: bool = False, repo_root: Path | None = None) -> ManifestReport:
    errors, warnings = [], []
    for k in REQUIRED_KEYS:
        if k not in m:
            errors.append(f"missing key: {k}")
    if errors:
        return ManifestReport(errors, warnings)
    if not str(m["match_id"]).strip():
        errors.append("match_id is empty")
    perm = m["video_use_permission"]
    if perm not in PERMISSION_VALUES:
        errors.append(f"video_use_permission must be one of {sorted(PERMISSION_VALUES)}")
    elif perm == "denied":
        errors.append("video_use_permission is 'denied': do not process this footage")
    elif perm == "pending":
        (errors if production else warnings).append("video_use_permission is 'pending'")
    fps = m["normalized_fps"]
    if not isinstance(fps, (int, float)) or fps <= 0:
        errors.append("normalized_fps must be a positive number")
    for dim in ("field_length_m", "field_width_m"):
        v = m[dim]
        if v is not None and (not isinstance(v, (int, float)) or v <= 0):
            errors.append(f"{dim} must be positive metres or null")
    if m["field_length_m"] and m["field_width_m"] and m["field_width_m"] > m["field_length_m"]:
        errors.append("field_width_m exceeds field_length_m: check orientation")
    pf = m.get("play_format")
    if pf is not None and pf not in PLAY_FORMATS:
        errors.append(f"play_format must be one of {sorted(PLAY_FORMATS)}")
    elif pf is None:
        warnings.append("play_format not set (7v7/9v9/11v11): needed before comparing cohorts")
    unresolved = [k for k in PRODUCTION_REQUIRED if m.get(k) is None]
    if unresolved:
        (errors if production else warnings).append(f"unresolved values (verify, never infer): {unresolved}")
    if m["home_team_id"] is not None and m["home_team_id"] == m["away_team_id"]:
        errors.append("home_team_id equals away_team_id")
    if repo_root is not None:
        vp = (repo_root / m["normalized_video"])
        if not vp.exists():
            errors.append(f"normalized_video not found: {m['normalized_video']}")
    return ManifestReport(errors, warnings)
