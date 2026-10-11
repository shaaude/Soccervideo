"""Review clips for candidate events (Step 10, PR2).

All timestamps are validated before any clip is cut, so a bad row fails the stage without
leaving a partial set. Each clip is probed after encoding to confirm it is playable.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd

from ..provenance import video_summary

CLIP_INDEX_COLUMNS = ["event_id", "clip_path", "start_seconds", "end_seconds", "event_time_s", "clip_duration_s"]
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_name(event_id: str) -> str:
    name = _SAFE.sub("_", str(event_id))
    if not name or name.startswith("."):
        raise ValueError(f"Unusable event_id for a filename: {event_id!r}")
    return name


def plan_clips(events: pd.DataFrame, duration: float, before: float, after: float,
               time_offset: float = 0.0, skip_outside: bool = False, skipped: list | None = None) -> list[dict]:
    if before < 0 or after <= 0:
        raise ValueError("Clip padding must be nonnegative before, positive after")
    if not {"event_id", "video_time_s"}.issubset(events.columns):
        raise ValueError("Missing event_id / video_time_s columns")
    if events["event_id"].duplicated().any():
        raise ValueError("Duplicate event_id in events file")
    bad = []
    plan = []
    for row in events.itertuples(index=False):
        try:
            ts = float(row.video_time_s) + time_offset
        except (TypeError, ValueError):
            bad.append(f"{row.event_id}: non-numeric time")
            continue
        if not 0 <= ts <= duration:
            if skip_outside and skipped is not None:
                skipped.append(str(row.event_id))
                continue
            bad.append(f"{row.event_id}: {ts}s outside 0-{duration:.3f}s")
            continue
        start, end = max(0.0, ts - before), min(duration, ts + after)
        plan.append({"event_id": str(row.event_id), "name": safe_name(row.event_id),
                     "start_seconds": round(start, 3), "end_seconds": round(end, 3), "event_time_s": ts})
    if bad:
        raise ValueError("Events outside source video (no clips written): " + "; ".join(bad))
    names = [p["name"] for p in plan]
    if len(set(names)) != len(names):
        raise ValueError("Two event_ids map to the same clip filename")
    return plan


def make_highlights(video: Path, events: pd.DataFrame, out_dir: Path, before=4.0, after=6.0,
                    time_offset: float = 0.0, skip_outside: bool = False) -> pd.DataFrame:
    """Cut review clips. `time_offset` maps event times into another feed of the same match
    (e.g. Veo follow-cam = panorama time + offset); with `skip_outside`, events the other feed does not
    cover are listed in `skipped_events.txt` instead of failing the stage."""
    video, out_dir = Path(video), Path(out_dir)
    duration = video_summary(video)["duration_seconds"]
    skipped: list[str] = []
    plan = plan_clips(events, duration, before, after, time_offset, skip_outside, skipped)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in plan:
        target = out_dir / f"{c['name']}.mp4"
        tmp = target.with_suffix(".tmp.mp4")
        # Input seeking + re-encode is frame-accurate and avoids decoding from the start.
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{c['start_seconds']:.3f}",
               "-i", str(video), "-t", f"{c['end_seconds'] - c['start_seconds']:.3f}", "-map", "0:v:0", "-an",
               "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(tmp)]
        subprocess.run(cmd, check=True)
        got = video_summary(tmp)["duration_seconds"]
        if got <= 0:
            raise RuntimeError(f"Clip for {c['event_id']} is not playable")
        tmp.replace(target)
        rows.append({"event_id": c["event_id"], "clip_path": str(target), "start_seconds": c["start_seconds"],
                     "end_seconds": c["end_seconds"], "event_time_s": c["event_time_s"],
                     "clip_duration_s": round(got, 3)})
    index = pd.DataFrame(rows, columns=CLIP_INDEX_COLUMNS)
    index.to_csv(out_dir / "clip_index.csv", index=False)
    if skip_outside:
        (out_dir / "skipped_events.txt").write_text("\n".join(skipped) + ("\n" if skipped else ""))
    return index
