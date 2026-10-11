"""Veo link ingest: URL safety, resumable download, dual-feed manifest, follow-cam clips, tiled detection."""
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest
import yaml

from soccer_vision.detection.export import RawDetections
from soccer_vision.detection.tiling import TiledPredictor, tile_origins
from soccer_vision.ingest import IngestError, download, ingest_url, validate_url
from soccer_vision.provenance import sha256_file
from soccer_vision.veo import feed_frames, run_veo, set_offset
from conftest import REPO, FakePredictor

REC = "67766a5f-59d6-4021-ae4c-2e1f8058aa3f"
PANO = f"/{REC}/panorama/transcode-520becdc-1ef6-4642-b320-19e68d5b620b.mp4"
STD = f"/{REC}/standard/machine/68ffd7d8/video.mp4"
OTHER = "/11111111-2222-3333-4444-555555555555/standard/machine/aa/video.mp4"


def _video(path, size, seconds=12):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc2=duration={seconds}:size={size}:rate=30", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("veo")
    files = {PANO: _video(d / "pano.mp4", "1920x360"), STD: _video(d / "std.mp4", "640x360"),
             OTHER: _video(d / "other.mp4", "640x360")}
    state = {"drop_once": set()}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/redirect.mp4":
                self.send_response(302); self.send_header("Location", "https://evil.example.com/x.mp4"); self.end_headers(); return
            if path == "/expired.mp4":
                self.send_response(403); self.end_headers(); return
            if path == "/page.mp4":
                body = b"<html>sign in</html>"
                self.send_response(200); self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            f = files.get(path)
            if f is None:
                self.send_response(404); self.end_headers(); return
            data = f.read_bytes()
            start = 0
            rng = self.headers.get("Range")
            if rng:
                start = int(rng.split("=")[1].split("-")[0])
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data) - start))
            self.end_headers()
            if path in state["drop_once"] and not rng:
                state["drop_once"].discard(path)
                self.wfile.write(data[: len(data) // 3])  # connection drops mid-file
                self.wfile.flush()
                self.connection.close()
                return
            self.wfile.write(data[start:])

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{srv.server_port}"
    yield {"base": base, "files": files, "state": state}
    srv.shutdown()


HOSTS = ("127.0.0.1",)


def test_url_rules():
    ok = validate_url(f"https://c.veocdn.com{PANO}", allow_private=True)  # no DNS lookup in tests
    assert ok.recording_id == REC and ok.feed_hint == "panorama" and ok.provider == "veo"
    assert validate_url(f"https://c.veocdn.com{STD}?token=abc", allow_private=True).redacted_url == f"https://c.veocdn.com{STD}"
    for bad, msg in [(f"http://c.veocdn.com{PANO}", "https"), (f"https://evil.com{PANO}", "allowlist"),
                     (f"https://u:p@c.veocdn.com{PANO}", "credentials"), ("https://c.veocdn.com/x/page.html", ".mp4")]:
        with pytest.raises(IngestError, match=msg):
            validate_url(bad)
    with pytest.raises(IngestError, match="non-public"):
        validate_url("https://localhost/x.mp4", hosts=("localhost",))


def test_resume_after_dropped_connection(server, tmp_path):
    server["state"]["drop_once"].add(PANO)
    r = download(server["base"] + PANO, tmp_path / "p.mp4", hosts=HOSTS, allow_private=True, backoff=0)
    assert r["resumed"] and r["attempts"] == 2
    assert r["sha256"] == sha256_file(server["files"][PANO])
    assert not (tmp_path / "p.mp4.part").exists()


@pytest.mark.parametrize("path,msg", [("/redirect.mp4", "allowlist"), ("/expired.mp4", "expired"),
                                      ("/page.mp4", "Content-Type")])
def test_download_refusals(server, tmp_path, path, msg):
    with pytest.raises(IngestError, match=msg):
        download(server["base"] + path, tmp_path / "x.mp4", hosts=HOSTS, allow_private=True, backoff=0)
    assert not (tmp_path / "x.mp4").exists()


def test_dual_feed_ingest_manifest_and_redaction(server, tmp_path):
    repo = tmp_path / "repo"
    res = run_veo(repo, "veo1", server["base"] + PANO + "?sig=SECRET", server["base"] + STD, "granted",
                  allow_private=True, hosts=HOSTS, log=lambda *a: None)
    m = res["manifest"]
    assert m["analysis_feed"] == "panorama" and m["review_feed"] == "standard"
    assert m["normalized_video"] == "data/normalized/veo1-panorama.mp4" and m["normalized_fps"] == 25
    assert m["feeds"]["panorama"]["normalized"]["width"] == 1920      # panorama keeps native width
    assert m["camera_type"] == "veo_panorama_stitched_static" and m["video_use_permission"] == "granted"
    assert m["standard_offset_seconds"] is None and m["standard_offset_verified"] is False   # never inferred
    assert m["feed_duration_difference_seconds"] == pytest.approx(0, abs=0.1)
    stored = (repo / "config/veo1.json").read_text() + (repo / "data/raw/veo1-panorama.ingest.json").read_text()
    assert "SECRET" not in stored and m["feeds"]["panorama"]["source"]["query_redacted"] is True
    assert (repo / "data/raw/veo1-panorama-original.mp4").exists()     # original preserved
    # a follow-cam from a different recording cannot be attached to this match
    with pytest.raises(IngestError, match="not the same match"):
        ingest_url(repo, "veo1", server["base"] + OTHER, "standard", allow_private=True, hosts=HOSTS)
    with pytest.raises(IngestError, match="looks like the panorama"):
        ingest_url(repo, "veo2", server["base"] + PANO, "standard", allow_private=True, hosts=HOSTS)
    with pytest.raises(IngestError, match="denied"):
        ingest_url(repo, "veo3", server["base"] + PANO, permission="denied", allow_private=True, hosts=HOSTS)
    out = feed_frames(repo, "veo1", 5.0, 0.0, tmp_path / "check.png")
    assert out.exists() and out.stat().st_size > 1000


def test_pipeline_cuts_follow_cam_clips_only_after_offset_is_verified(server, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    run_veo(repo, "veo1", server["base"] + PANO, server["base"] + STD, "granted",
            allow_private=True, hosts=HOSTS, log=lambda *a: None)
    (repo / "config/classmaps").mkdir(parents=True)
    shutil.copy(REPO / "config/classmaps/soccernet_yolov8_smoke.yaml", repo / "config/classmaps")
    (repo / "models").mkdir()
    (repo / "models/fake.pt").write_bytes(b"w")
    cfg = yaml.safe_load((REPO / "config/pipeline.veo.yaml").read_text())
    cfg["detection"].update({"class_map": "config/classmaps/soccernet_yolov8_smoke.yaml",
                             "player_weights": "models/fake.pt", "ball_weights": None, "annotated_video": False})
    (repo / "config/pipeline.yaml").write_text(yaml.safe_dump(cfg))
    from test_pipeline import NAMES, script
    monkeypatch.setattr("soccer_vision.detection.export.UltralyticsPredictor",
                        lambda *a, **k: FakePredictor(NAMES, script()))
    from soccer_vision.pipeline.runner import process
    st = process(repo, "veo1", "data/normalized/veo1-panorama.mp4", repo / "config/pipeline.yaml")
    assert "follow-cam clips NOT cut" in st["stages"]["highlights"]["notes"]
    set_offset(repo / "config/veo1.json", 0.0, "reviewer-1")
    st2 = process(repo, "veo1", "data/normalized/veo1-panorama.mp4", repo / "config/pipeline.yaml")
    assert "3 follow-cam clips" in st2["stages"]["highlights"]["notes"]
    assert st2["run_id"] == st["run_id"]  # only highlights re-ran
    idx = pd.read_csv(repo / "data/processed/veo1" / st["run_id"] / "highlights_review/clip_index.csv")
    assert len(idx) == 3


class _Grid:
    """Fake detector: reports every player whose centre lies in the tile it is shown. The frame encodes
    each pixel's x coordinate in its colour, so the detector can tell where the tile came from."""
    names = {0: "Player"}

    def __init__(self, players):
        self.players = np.array(players, float)

    def __call__(self, tile):
        x0 = int(tile[0, 0, 0]) + 256 * int(tile[0, 0, 1])
        h, w = tile.shape[:2]
        b = self.players - np.array([x0, 0, x0, 0])
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        m = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
        clipped = np.clip(b[m], 0, [w, h, w, h])
        return RawDetections(clipped, np.full(int(m.sum()), 0.9), np.zeros(int(m.sum()), int))


def test_tiled_inference_finds_all_and_merges_overlap_duplicates():
    assert tile_origins(1000, 1280, 0.2) == [0]
    assert tile_origins(3000, 1280, 0.2) == [0, 1024, 1720]
    frame = np.zeros((360, 3000, 3), np.uint8)
    xs = np.arange(3000)
    frame[:, :, 0], frame[:, :, 1] = xs % 256, xs // 256
    # the middle player sits where tiles 1 and 2 overlap, so both tiles report it
    players = [[100, 100, 120, 160], [1730, 100, 1760, 170], [2900, 200, 2920, 260]]
    det = TiledPredictor(_Grid(players), 1280, 0.2)(frame)
    assert len(det.confidence) == 3
    assert np.allclose(sorted(det.xyxy[:, 0]), [100, 1730, 2900])
    small = np.zeros((360, 640, 3), np.uint8)
    assert len(TiledPredictor(_Grid([[10, 10, 20, 40]]), 1280)(small).confidence) == 1  # narrow frame: one pass


def test_api_video_from_url(server, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from soccer_vision.api.app import create_app
    from soccer_vision.db.models import Base
    from soccer_vision.db.session import make_engine
    monkeypatch.setenv("SOCCER_API_KEYS", "adm:admin:alice,rev:reviewer:bob")
    monkeypatch.setenv("SOCCER_ALLOWED_VIDEO_HOSTS", "127.0.0.1")
    eng = make_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    Base.metadata.create_all(eng)
    repo = tmp_path / "repo"
    repo.mkdir()
    app = create_app(eng, repo, executor=lambda fn, *a: fn(*a))
    c = TestClient(app)
    H = {"X-API-Key": "adm"}
    assert c.post("/api/v1/matches", headers=H, json={"match_id": "veo1"}).status_code == 201
    body = {"url": server["base"] + PANO, "video_use_permission": "granted"}
    assert c.post("/api/v1/matches/veo1/videos/from-url", headers={"X-API-Key": "rev"}, json=body).status_code == 403
    r = c.post("/api/v1/matches/veo1/videos/from-url", headers=H, json=body)
    assert r.status_code == 422 and "https" in r.json()["detail"]  # plain-http/private targets refused by default
    app.state.allow_private_urls = True  # test server only
    r = c.post("/api/v1/matches/veo1/videos/from-url", headers=H, json=body)
    assert r.status_code == 202 and r.json()["feed"] == "panorama"
    j = c.get(f"/api/v1/jobs/{r.json()['job_id']}", headers=H).json()
    assert j["kind"] == "ingest" and j["state"] == "completed" and j["video_id"] == "veo1-panorama"
    r = c.post("/api/v1/matches/veo1/videos/from-url", headers=H,
               json={"url": "https://evil.example.com/a.mp4", "video_use_permission": "granted"})
    assert r.status_code == 422 and "allowlist" in r.json()["detail"]
    # the registered panorama can now be processed through the normal endpoint
    assert c.post("/api/v1/matches/veo1/process", headers=H, json={"video_id": "veo1-panorama"}).status_code == 202


def test_recorded_permission_is_applied_without_flag(server, tmp_path):
    repo = tmp_path / "repo"
    (repo / "config").mkdir(parents=True)
    shutil.copy(REPO / "config/video_permissions.yaml", repo / "config")
    res = ingest_url(repo, "veo9", server["base"] + PANO, allow_private=True, hosts=HOSTS)
    m = res["manifest"]
    assert m["video_use_permission"] == "granted"
    assert m["video_use_permission_record"]["granted_by"] == "team Veo account owner"
    # an unlisted recording stays pending
    res2 = ingest_url(repo, "veo10", server["base"] + OTHER, allow_private=True, hosts=HOSTS)
    assert res2["manifest"]["video_use_permission"] == "pending"


def test_follow_cam_only_link(server, tmp_path):
    repo = tmp_path / "repo"
    (repo / "config").mkdir(parents=True)
    shutil.copy(REPO / "config/video_permissions.yaml", repo / "config")
    msgs = []
    res = run_veo(repo, "oct-match", None, server["base"] + STD, None, clip_seconds=5,
                  allow_private=True, hosts=HOSTS, log=msgs.append)
    m = res["manifest"]
    assert m["analysis_feed"] == m["review_feed"] == "standard"
    assert m["normalized_video"] == "data/normalized/oct-match-standard.mp4"
    assert m["video_use_permission"] == "granted"          # from config/video_permissions.yaml
    assert any("no panorama feed" in x for x in msgs)
