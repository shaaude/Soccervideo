"""Video normalization (Step 3.3): constant FPS, bounded width, H.264, no audio.

The original file is never modified; provenance for both files is written next to the output.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .provenance import sha256_file, video_summary


def normalize_video(original: Path, output: Path, fps: int = 25, max_width: int = 1920,
                    crf: int = 18, clip_seconds: float | None = None, start_seconds: float = 0.0) -> dict:
    original, output = Path(original), Path(output)
    if not original.exists():
        raise FileNotFoundError(original)
    if output.resolve() == original.resolve():
        raise ValueError("Output must differ from the original (originals are preserved)")
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start_seconds:
        cmd += ["-ss", f"{start_seconds:.3f}"]
    cmd += ["-i", str(original)]
    if clip_seconds:
        # Re-encoding (not stream copy) keeps the cut frame-accurate.
        cmd += ["-t", f"{clip_seconds:.3f}"]
    cmd += ["-map", "0:v:0", "-an",
            "-vf", f"fps={fps},scale='min({max_width},iw)':-2",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    subprocess.run(cmd, check=True)
    summary = video_summary(output)
    if not summary["constant_frame_rate"] or abs((summary["avg_frame_rate"] or 0) - fps) > 1e-3:
        raise RuntimeError(f"Normalized video is not constant {fps} FPS: {summary}")
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "original": {"path": str(original), "sha256": sha256_file(original), **_safe_summary(original)},
        "normalized": {"path": str(output), "sha256": sha256_file(output), **summary},
        "parameters": {"fps": fps, "max_width": max_width, "crf": crf,
                       "clip_seconds": clip_seconds, "start_seconds": start_seconds},
        "ffmpeg_command": cmd,
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(record, indent=2))
    return record


def _safe_summary(path):
    try:
        return video_summary(path)
    except Exception as exc:  # original may have odd streams; record, don't fail
        return {"probe_error": str(exc)}
