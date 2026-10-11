import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def make_tracks_balls(owner_by_frame, n_frames, match="m1", players=((1, 105), (2, 305)), fps=25.0):
    """Two static players; ball at a player's feet when owner_by_frame(f) names them, mid-air (far) when 'air',
    absent when None."""
    tracks, balls = [], []
    for f in range(n_frames):
        for tid, x in players:
            tracks.append({"match_id": match, "frame_index": f, "time_s": f / fps, "track_id": tid, "class_id": 2,
                           "confidence": 0.9, "x1": x - 5, "y1": 100, "x2": x + 5, "y2": 200,
                           "foot_x": x, "foot_y": 200})
        o = owner_by_frame(f)
        if o is None:
            continue
        cx = dict(players).get(o, 205) if o != "air" else 205
        cy = 200 if o != "air" else 120
        balls.append({"match_id": match, "frame_index": f, "time_s": f / fps, "confidence": 0.9,
                      "x1": cx - 3, "y1": cy - 3, "x2": cx + 3, "y2": cy + 3, "center_x": cx, "center_y": cy})
    return pd.DataFrame(tracks), pd.DataFrame(balls)


@pytest.fixture
def synthetic_video(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    p = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=12:size=320x240:rate=25",
                    "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


class FakePredictor:
    """Deterministic stand-in for a detector: returns scripted boxes per call."""

    def __init__(self, names, script):
        self.names = names
        self.script = script  # list of (xyxy, conf, cls) per frame
        self.i = 0

    def __call__(self, frame):
        from soccer_vision.detection.export import RawDetections
        if self.i >= len(self.script):
            return RawDetections.empty()
        xyxy, conf, cls = self.script[self.i]
        self.i += 1
        return RawDetections(np.array(xyxy, float).reshape(-1, 4), np.array(conf, float), np.array(cls, int))
