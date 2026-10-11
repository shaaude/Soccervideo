"""Provenance helpers: checksums, ffprobe metadata, code revision, environment locks."""
from __future__ import annotations

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def ffprobe(path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(out.stdout)


def video_summary(path) -> dict:
    """Return width/height/fps/duration and whether the stream looks constant-frame-rate."""
    info = ffprobe(path)
    vids = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not vids:
        raise ValueError(f"{path}: no video stream")
    v = vids[0]
    r = Fraction(v["r_frame_rate"]) if v.get("r_frame_rate", "0/0") != "0/0" else None
    a = Fraction(v["avg_frame_rate"]) if v.get("avg_frame_rate", "0/0") != "0/0" else None
    return {
        "width": int(v["width"]), "height": int(v["height"]),
        "codec": v.get("codec_name"),
        "r_frame_rate": float(r) if r else None,
        "avg_frame_rate": float(a) if a else None,
        "constant_frame_rate": bool(r and a and abs(float(r) - float(a)) < 1e-3),
        "nb_frames": int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else None,
        "duration_seconds": float(info["format"]["duration"]),
        "size_bytes": int(info["format"].get("size", 0)),
    }


def code_revision() -> str:
    """Git HEAD (with -dirty marker) if available, else a hash of the package source files."""
    try:
        root = subprocess.run(["git", "-C", str(PACKAGE_ROOT), "rev-parse", "--show-toplevel"],
                              capture_output=True, text=True, check=True).stdout.strip()
        head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", "src", "scripts", "config"],
                               capture_output=True, text=True, check=True).stdout.strip()
        if not dirty:
            return head
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    h = hashlib.sha256()
    for p in sorted(PACKAGE_ROOT.rglob("*.py")):
        h.update(p.relative_to(PACKAGE_ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return "src-" + h.hexdigest()[:16]
