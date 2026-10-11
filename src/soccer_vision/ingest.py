"""Ingest match video from a URL (e.g. a Veo recording link) into the pipeline's data layout.

    soccer-vision ingest --match-id m42 --feed panorama --permission granted \
        --url https://c.veocdn.com/<recording>/panorama/transcode-<id>.mp4

What happens:
  1. The URL is validated: https only, host on the allowlist (default c.veocdn.com; override with
     SOCCER_ALLOWED_VIDEO_HOSTS), no embedded credentials, and the host must not resolve to a private,
     loopback or link-local address (so the API cannot be used to reach internal services).
  2. The file is streamed to data/raw/<match>-<feed>-original.mp4 with resume (HTTP Range) and retries,
     written to a .part file and renamed only when complete and the size matches Content-Length.
  3. sha256, ffprobe metadata and a provenance record are written. Any query string (signed-URL tokens)
     is redacted before it is stored anywhere.
  4. The file is normalized to constant FPS (the original is kept) and the match manifest is created or
     updated with a `feeds` section. The panorama, when present, becomes the analysis feed; the
     follow-cam ("standard") feed becomes the review feed.

Nothing is inferred silently: the follow-cam/panorama time offset stays null until a person verifies it.
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import shutil
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .manifest import load_manifest
from .normalize import normalize_video
from .provenance import sha256_file, video_summary

DEFAULT_ALLOWED_HOSTS = ("c.veocdn.com",)
FEEDS = ("panorama", "standard")
# Panorama frames are very wide; keep native detail (players are small). Follow-cam is ordinary HD.
DEFAULT_MAX_WIDTH = {"panorama": 4096, "standard": 1920}
CAMERA_TYPE = {"panorama": "veo_panorama_stitched_static", "standard": "veo_follow_cam_moving"}
_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


class IngestError(RuntimeError):
    pass


@dataclass
class SourceInfo:
    url: str               # full URL, used only for the request
    redacted_url: str      # stored in provenance / manifest
    host: str
    provider: str
    recording_id: str | None
    feed_hint: str | None


def allowed_hosts() -> tuple[str, ...]:
    env = os.environ.get("SOCCER_ALLOWED_VIDEO_HOSTS")
    return tuple(h.strip().lower() for h in env.split(",") if h.strip()) if env else DEFAULT_ALLOWED_HOSTS


def redact(url: str) -> str:
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.hostname or "", p.path, "", ""))


def parse_source(url: str) -> SourceInfo:
    p = urlsplit(url)
    host = (p.hostname or "").lower()
    rec, feed = None, None
    m = re.match(rf"^/({_UUID})/(panorama|standard)/", p.path, re.I)   # Veo CDN path layout
    if m:
        rec, feed = m.group(1).lower(), m.group(2).lower()
    return SourceInfo(url, redact(url), host, "veo" if host.endswith("veocdn.com") else "generic", rec, feed)


def validate_url(url: str, hosts: tuple[str, ...] | None = None, allow_private: bool = False) -> SourceInfo:
    p = urlsplit(url)
    if p.scheme != "https" and not (allow_private and p.scheme == "http"):
        raise IngestError("Only https:// video URLs are accepted")
    if p.username or p.password:
        raise IngestError("URLs with embedded credentials are not accepted")
    host = (p.hostname or "").lower()
    hosts = hosts or allowed_hosts()
    if host not in hosts:
        raise IngestError(f"Host '{host}' is not on the allowlist {list(hosts)} (set SOCCER_ALLOWED_VIDEO_HOSTS)")
    if not p.path.lower().endswith(".mp4"):
        raise IngestError("Expected a direct .mp4 file URL")
    if not allow_private:
        try:
            addrs = {ai[4][0] for ai in socket.getaddrinfo(host, p.port or 443, proto=socket.IPPROTO_TCP)}
        except socket.gaierror as e:
            # Behind an HTTPS proxy, names are often resolved only by the proxy. The host is still
            # restricted to the allowlist, so continue and let the proxy resolve it.
            if not (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")):
                raise IngestError(f"Cannot resolve {host}: {e}") from e
            addrs = set()
        for a in addrs:
            ip = ipaddress.ip_address(a)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                raise IngestError(f"{host} resolves to a non-public address ({a}); refusing")
    return parse_source(url)


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    """Re-validate every redirect target, so a redirect cannot lead to a non-allowlisted or private host."""

    def __init__(self, hosts, allow_private):
        self.hosts, self.allow_private = hosts, allow_private

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl, self.hosts, self.allow_private)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _request(url, start: int = 0, timeout: float = 60.0, hosts=None, allow_private=False):
    req = urllib.request.Request(url, headers={"User-Agent": "soccer-vision-ingest/0.1",
                                               **({"Range": f"bytes={start}-"} if start else {})})
    opener = urllib.request.build_opener(_CheckedRedirects(hosts, allow_private))
    return opener.open(req, timeout=timeout)


def download(url: str, dest: Path, max_bytes: int = 30 * 1024**3, retries: int = 4, chunk: int = 1 << 20,
             timeout: float = 60.0, progress=None, hosts=None, allow_private: bool = False,
             backoff: float = 2.0) -> dict:
    """Stream `url` to `dest` with resume. Returns {bytes, sha256, content_type, resumed, attempts}."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    attempts, resumed, total, ctype = 0, False, None, None
    while True:
        attempts += 1
        have = part.stat().st_size if part.exists() else 0
        try:
            with _request(url, have, timeout, hosts, allow_private) as r:
                status = r.status
                ctype = r.headers.get("Content-Type", "")
                if ctype and not (ctype.startswith("video/") or ctype in ("application/octet-stream", "binary/octet-stream")):
                    raise IngestError(f"Unexpected Content-Type '{ctype}' (expected a video file)")
                if have and status == 206:
                    resumed = True
                    m = re.match(r"bytes \d+-\d+/(\d+)", r.headers.get("Content-Range", ""))
                    total = int(m.group(1)) if m else None
                    mode = "ab"
                else:
                    if have:  # server ignored Range: start over
                        have = 0
                    cl = r.headers.get("Content-Length")
                    total = int(cl) if cl and cl.isdigit() else None
                    mode = "wb"
                if total is not None:
                    if total > max_bytes:
                        raise IngestError(f"File is {total / 1e9:.1f} GB, above the {max_bytes / 1e9:.0f} GB limit")
                    free = shutil.disk_usage(dest.parent).free
                    if total - have > free:
                        raise IngestError(f"Not enough disk space: need {(total - have) / 1e9:.1f} GB, "
                                          f"have {free / 1e9:.1f} GB")
                with open(part, mode) as f:
                    written = have
                    while True:
                        buf = r.read(chunk)
                        if not buf:
                            break
                        written += len(buf)
                        if written > max_bytes:
                            raise IngestError("Download exceeded the size limit")
                        f.write(buf)
                        if progress:
                            progress(written, total)
            size = part.stat().st_size
            if total is not None and size != total:
                raise ConnectionError(f"Incomplete download: {size} of {total} bytes")
            break
        except IngestError:
            raise
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404, 410):
                raise IngestError(f"HTTP {e.code} from server: the link may have expired or need sign-in") from e
            if attempts > retries:
                raise IngestError(f"HTTP {e.code} after {attempts} attempts") from e
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            if attempts > retries:
                raise IngestError(f"Download failed after {attempts} attempts: {e}") from e
        time.sleep(min(30.0, backoff ** attempts) if backoff else 0)
    os.replace(part, dest)
    return {"bytes": dest.stat().st_size, "sha256": sha256_file(dest), "content_type": ctype,
            "resumed": resumed, "attempts": attempts}


def recorded_permission(repo: Path, recording_id: str | None) -> dict | None:
    """Permission recorded in config/video_permissions.yaml for this recording, if any."""
    import yaml
    path = Path(repo) / "config" / "video_permissions.yaml"
    if not recording_id or not path.exists():
        return None
    entry = ((yaml.safe_load(path.read_text()) or {}).get("recordings") or {}).get(recording_id)
    if entry is None:
        return None
    if entry.get("permission") not in ("pending", "granted", "denied"):
        raise IngestError(f"config/video_permissions.yaml: invalid permission for {recording_id}")
    return {k: str(v) for k, v in entry.items() if k in ("permission", "granted_by", "basis", "recorded_at")}


def _manifest_template(match_id: str) -> dict:
    return {"match_id": match_id, "video_id": None, "season": None, "competition": None, "age_group": None,
            "division": None, "play_format": None, "home_team_id": None, "away_team_id": None,
            "match_date": None, "field_length_m": None, "field_width_m": None, "camera_id": None,
            "camera_type": None, "normalized_video": None, "normalized_fps": None,
            "match_clock_offset_seconds": None, "video_use_permission": "pending",
            "calibration_landmarks": None, "roster": None}


def update_manifest(path: Path, match_id: str, feed: str, record: dict, permission: str | None,
                    permission_record: dict | None = None) -> dict:
    m = load_manifest(path) if path.exists() else _manifest_template(match_id)
    if m["match_id"] != match_id:
        raise IngestError(f"{path} belongs to match {m['match_id']}")
    feeds = m.setdefault("feeds", {})
    rid = record["source"].get("recording_id")
    for other, info in feeds.items():
        orid = info.get("source", {}).get("recording_id")
        if other != feed and rid and orid and rid != orid:
            raise IngestError(f"{feed} feed is from recording {rid} but {other} is from {orid}: not the same match")
    feeds[feed] = record
    analysis = "panorama" if "panorama" in feeds else feed
    a = feeds[analysis]
    m.update({"analysis_feed": analysis, "review_feed": "standard" if "standard" in feeds else analysis,
              "video_id": f"{match_id}-{analysis}", "camera_id": f"veo-{analysis}" if a["source"]["provider"] == "veo" else analysis,
              "camera_type": CAMERA_TYPE.get(analysis, analysis), "normalized_video": a["normalized"]["path"],
              "original_video": a["original"]["path"], "normalized_fps": a["normalized"]["fps"]})
    m.setdefault("standard_offset_seconds", None)
    m.setdefault("standard_offset_verified", False)
    if permission:
        m["video_use_permission"] = permission
    if permission_record:
        m["video_use_permission_record"] = permission_record
    if "panorama" in feeds and "standard" in feeds:
        d = abs(feeds["panorama"]["normalized"]["duration_seconds"] - feeds["standard"]["normalized"]["duration_seconds"])
        m["feed_duration_difference_seconds"] = round(d, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2))
    tmp.replace(path)
    return m


def ingest_url(repo: Path, match_id: str, url: str, feed: str | None = None, permission: str | None = None,
               fps: int = 25, max_width: int | None = None, allow_private: bool = False,
               hosts: tuple[str, ...] | None = None, clip_seconds: float | None = None, progress=None) -> dict:
    if permission == "denied":
        raise IngestError("video_use_permission is 'denied': not downloading")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", match_id):
        raise IngestError("match_id may contain letters, digits, '-' and '_' only")
    src = validate_url(url, hosts, allow_private)
    repo = Path(repo)
    perm_rec = recorded_permission(repo, src.recording_id)
    if permission is None and perm_rec:
        permission = perm_rec["permission"]
    elif permission is not None:
        perm_rec = {"permission": permission, "basis": "given on the command line / API request",
                    "recorded_at": datetime.now(timezone.utc).date().isoformat()}
    if permission == "denied":
        raise IngestError("video_use_permission is 'denied' for this recording: not downloading")
    feed = feed or src.feed_hint
    if feed not in FEEDS:
        raise IngestError("Say which feed this is with --feed panorama|standard (could not tell from the URL)")
    if src.feed_hint and src.feed_hint != feed:
        raise IngestError(f"URL looks like the {src.feed_hint} feed but --feed {feed} was given")
    raw = repo / "data" / "raw" / f"{match_id}-{feed}-original.mp4"
    norm = repo / "data" / "normalized" / f"{match_id}-{feed}.mp4"
    meta = repo / "data" / "raw" / f"{match_id}-{feed}.ingest.json"
    mpath = repo / "config" / f"{match_id}.json"
    # Check consistency BEFORE downloading gigabytes.
    if mpath.exists() and src.recording_id:
        for other, info in (load_manifest(mpath).get("feeds") or {}).items():
            orid = info.get("source", {}).get("recording_id")
            if other != feed and orid and orid != src.recording_id:
                raise IngestError(f"{feed} feed is from recording {src.recording_id} but {other} is from {orid}: "
                                  "not the same match")
    if raw.exists():
        prev = json.loads(meta.read_text())["source"]["url"] if meta.exists() else None
        if prev != src.redacted_url:
            raise IngestError(f"{raw.name} already exists from a different source ({prev}); "
                              "use another match_id or remove the old files")
    started = datetime.now(timezone.utc).isoformat()
    if raw.exists():
        dl = {"bytes": raw.stat().st_size, "sha256": sha256_file(raw), "reused_existing_download": True}
    else:
        dl = download(src.url, raw, progress=progress, hosts=hosts, allow_private=allow_private)
    try:
        original = video_summary(raw)
    except Exception as e:
        raw.rename(raw.with_suffix(".invalid"))
        raise IngestError(f"Downloaded file is not a readable video: {e}") from e
    width = max_width or DEFAULT_MAX_WIDTH[feed]
    rec = normalize_video(raw, norm, fps=fps, max_width=width, clip_seconds=clip_seconds)
    record = {
        "source": {"url": src.redacted_url, "query_redacted": "?" in url, "provider": src.provider,
                   "recording_id": src.recording_id, "feed": feed},
        "downloaded_at": started,
        "original": {"path": str(raw.relative_to(repo)), "sha256": dl["sha256"], "bytes": dl["bytes"],
                     "width": original["width"], "height": original["height"],
                     "fps": original["avg_frame_rate"], "duration_seconds": original["duration_seconds"]},
        "normalized": {"path": str(norm.relative_to(repo)), "sha256": rec["normalized"]["sha256"],
                       "width": rec["normalized"]["width"], "height": rec["normalized"]["height"],
                       "fps": rec["normalized"]["avg_frame_rate"],
                       "duration_seconds": rec["normalized"]["duration_seconds"],
                       "max_width": width, "clip_seconds": clip_seconds},
    }
    meta.write_text(json.dumps({**record, "download": dl}, indent=2))
    manifest = update_manifest(mpath, match_id, feed, record, permission, perm_rec)
    return {"feed": feed, "record": record, "manifest": manifest}
