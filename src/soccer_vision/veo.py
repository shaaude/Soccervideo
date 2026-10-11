"""Veo-specific helpers: one-shot ingest+process, follow-cam offset, and frame tools for people.

Veo gives two feeds of the same recording:
  panorama  stitched ~180-degree view from a static camera: every player in frame all match. Used for
            analysis (detection, tracking, candidates, calibration once for the whole match).
  standard  the AI follow-cam crop that pans and zooms with play. Used for review clips (easier to watch).
            Not used for analysis: the moving view breaks tracking and fixed-landmark calibration.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .ingest import IngestError, ingest_url
from .manifest import load_manifest


def run_veo(repo: Path, match_id: str, panorama_url: str | None, standard_url: str | None,
            permission: str | None, config: Path | None = None, process: bool = False,
            clip_seconds: float | None = None, allow_private: bool = False, hosts=None, log=print) -> dict:
    if not panorama_url and not standard_url:
        raise IngestError("Give at least one of --panorama / --standard")
    out = {}
    for feed, url in (("panorama", panorama_url), ("standard", standard_url)):
        if url:
            log(f"Downloading {feed} feed …")
            res = ingest_url(repo, match_id, url, feed, permission, clip_seconds=clip_seconds,
                             allow_private=allow_private, hosts=hosts,
                             progress=_progress_printer(log, feed))
            n = res["record"]["normalized"]
            log(f"  {feed}: {n['width']}x{n['height']} @ {n['fps']:.0f} fps, {n['duration_seconds'] / 60:.1f} min")
            out[feed] = res["record"]
    manifest = load_manifest(Path(repo) / "config" / f"{match_id}.json")
    out["manifest"] = manifest
    if not panorama_url:
        log("WARNING: no panorama feed. The follow-cam pans and zooms, so tracking will fragment and "
            "pitch coordinates are unavailable. Add the panorama link for analysis.")
    if process:
        from .pipeline.runner import process as run_process, summarize
        cfg = config or Path(repo) / "config" / "pipeline.veo.yaml"
        state = run_process(repo, match_id, manifest["normalized_video"], cfg)
        log(summarize(state).to_string(index=False))
        out["run_id"] = state["run_id"]
    return out


def _progress_printer(log, feed):
    last = {"pct": -10}

    def p(done, total):
        if total:
            pct = int(100 * done / total)
            if pct >= last["pct"] + 10:
                last["pct"] = pct
                log(f"  {feed}: {pct}% of {total / 1e9:.2f} GB")
    return p


def set_offset(manifest_path: Path, offset_seconds: float, verified_by: str) -> dict:
    """Record a person-verified offset: follow-cam time = panorama time + offset."""
    if not verified_by.strip():
        raise ValueError("verified_by is required (who checked the two frames match)")
    m = load_manifest(manifest_path)
    m.update({"standard_offset_seconds": float(offset_seconds), "standard_offset_verified": True,
              "standard_offset_verified_by": verified_by.strip(),
              "standard_offset_verified_at": datetime.now(timezone.utc).isoformat()})
    Path(manifest_path).write_text(json.dumps(m, indent=2))
    return m


def _frame(video: Path, t: float, out: Path, width: int | None = None):
    vf = [f"scale={width}:-2"] if width else []
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video),
           "-frames:v", "1"] + (["-vf", ",".join(vf)] if vf else []) + [str(out)]
    subprocess.run(cmd, check=True)


def feed_frames(repo: Path, match_id: str, t: float, offset: float, out: Path, width: int = 1600) -> Path:
    """Stack the panorama frame at t over the follow-cam frame at t+offset, to check an offset by eye."""
    m = load_manifest(Path(repo) / "config" / f"{match_id}.json")
    feeds = m.get("feeds") or {}
    if not {"panorama", "standard"} <= set(feeds):
        raise ValueError("Both feeds must be ingested first")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pano, std = out.with_suffix(".pano.png"), out.with_suffix(".std.png")
    _frame(Path(repo) / feeds["panorama"]["normalized"]["path"], t, pano, width)
    _frame(Path(repo) / feeds["standard"]["normalized"]["path"], t + offset, std, width)
    label = lambda s: f"drawtext=text='{s}':x=10:y=10:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.6"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(pano), "-i", str(std),
           "-filter_complex",
           f"[0]{label(f'panorama t={t:.2f}s')}[a];[1]{label(f'follow-cam t={t + offset:.2f}s (offset {offset:+.2f})')}[b];"
           "[a][b]vstack=inputs=2", str(out)]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:  # ffmpeg without drawtext: stack unlabeled
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(pano), "-i", str(std),
                        "-filter_complex", "[0][1]vstack=inputs=2", str(out)], check=True)
    pano.unlink(missing_ok=True)
    std.unlink(missing_ok=True)
    return out


def landmark_frame(video: Path, t: float, out: Path, grid: int = 100) -> Path:
    """Full-resolution frame with a labelled pixel grid, for reading off landmark image coordinates."""
    import cv2
    tmp = Path(out).with_suffix(".raw.png")
    _frame(Path(video), t, tmp)
    img = cv2.imread(str(tmp))
    tmp.unlink(missing_ok=True)
    h, w = img.shape[:2]
    for x in range(0, w, grid):
        cv2.line(img, (x, 0), (x, h - 1), (0, 255, 255) if x % (grid * 5) == 0 else (0, 160, 160), 1)
        if x % (grid * 5) == 0:
            cv2.putText(img, str(x), (x + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    for y in range(0, h, grid):
        cv2.line(img, (0, y), (w - 1, y), (0, 255, 255) if y % (grid * 5) == 0 else (0, 160, 160), 1)
        if y % (grid * 5) == 0:
            cv2.putText(img, str(y), (3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(str(out), img)
    return Path(out)
