"""Video normalization (Step 3.3): constant FPS, bounded width, H.264, no audio.

The original file is never modified; provenance for both files is written next to the output.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .provenance import sha256_file, video_summary


LAYOUTS = ("as_is", "stacked_halves")


def find_stack_seam(video: Path, samples: int = 5) -> int:
    """Row where the two halves of a vertically stacked panorama meet (largest brightness jump near the
    middle, agreed across sampled frames). Veo's app shows the panorama this way: left half on top,
    right half below."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    rows = []
    for k in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (k + 0.5) / samples))
        ok, f = cap.read()
        if not ok:
            continue
        h = f.shape[0]
        g = f.astype(float).mean(axis=(1, 2))
        lo, hi = int(h * 0.4), int(h * 0.6)
        d = np.abs(np.diff(g[lo:hi]))
        rows.append(lo + int(np.argmax(d)) + 1)
    cap.release()
    if not rows:
        raise ValueError(f"Cannot read frames from {video}")
    seam = int(np.median(rows))
    if max(rows) - min(rows) > 3:
        raise ValueError(f"No consistent seam between stacked halves (candidates {rows}); "
                         "is this really a stacked panorama?")
    return seam


def normalize_video(original: Path, output: Path, fps: int = 25, max_width: int = 1920,
                    crf: int = 18, clip_seconds: float | None = None, start_seconds: float = 0.0,
                    layout: str = "as_is", seam_row: int | None = None) -> dict:
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
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    if layout == "stacked_halves":
        s = seam_row or find_stack_seam(original)
        s = min(s, video_summary(original)["height"] - s)   # both halves must have the same height
        # top half = left of the pitch, bottom half = right; put them side by side at equal height
        vf = (f"[0:v:0]split[a][b];[a]crop=iw:{s}:0:0[l];[b]crop=iw:{s}:0:{s}[r];"
              f"[l][r]hstack,fps={fps},scale='min({max_width},iw)':-2[v]")
        cmd += ["-filter_complex", vf, "-map", "[v]", "-an"]
    else:
        cmd += ["-map", "0:v:0", "-an", "-vf", f"fps={fps},scale='min({max_width},iw)':-2"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
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
                       "clip_seconds": clip_seconds, "start_seconds": start_seconds,
                       "layout": layout, "seam_row": s if layout == "stacked_halves" else None},
        "ffmpeg_command": cmd,
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(record, indent=2))
    return record


def _safe_summary(path):
    try:
        return video_summary(path)
    except Exception as exc:  # original may have odd streams; record, don't fail
        return {"probe_error": str(exc)}
