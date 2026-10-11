import numpy as np
import pandas as pd
import pytest

from soccer_vision.detection.classmap import ClassMapConfig, ClassMapError, ModelClassMap
from soccer_vision.detection.export import ExportConfig, export_detections
from soccer_vision.events.highlights import make_highlights
from soccer_vision.provenance import video_summary
from soccer_vision.schemas import read_balls, read_tracks, validate_balls, validate_tracks
from conftest import REPO, FakePredictor

RF_NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}


def test_clips_inside_source_and_playable(synthetic_video, tmp_path):
    ev = pd.DataFrame({"event_id": ["m-1", "m-2"], "video_time_s": [1.0, 11.5]})
    idx = make_highlights(synthetic_video, ev, tmp_path / "clips")
    assert list(idx.start_seconds) == [0.0, 7.5]
    assert idx.end_seconds.max() <= video_summary(synthetic_video)["duration_seconds"] + 1e-6
    assert (idx.clip_duration_s > 0).all()
    assert (tmp_path / "clips" / "clip_index.csv").exists()


def test_bad_timestamp_fails_before_any_clip_is_written(synthetic_video, tmp_path):
    ev = pd.DataFrame({"event_id": ["ok", "bad"], "video_time_s": [2.0, 99.0]})
    with pytest.raises(ValueError, match="outside"):
        make_highlights(synthetic_video, ev, tmp_path / "clips")
    assert not (tmp_path / "clips").exists() or not list((tmp_path / "clips").iterdir())


def test_event_id_cannot_escape_output_dir(synthetic_video, tmp_path):
    ev = pd.DataFrame({"event_id": ["../../etc/x"], "video_time_s": [1.0]})
    with pytest.raises(ValueError, match="Unusable event_id"):
        make_highlights(synthetic_video, ev, tmp_path / "clips")
    ev = pd.DataFrame({"event_id": ["a/b"], "video_time_s": [1.0]})
    idx = make_highlights(synthetic_video, ev, tmp_path / "clips")
    assert idx.clip_path[0].endswith("clips/a_b.mp4")


def _single_model_map():
    # one model detects people and the ball (as in the smoke-test checkpoint)
    return ClassMapConfig("single", ModelClassMap(RF_NAMES, {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}), None)


def _frames(n):
    return ((i, np.zeros((240, 320, 3), np.uint8)) for i in range(n))


def _script(n):
    s = []
    for i in range(n):
        s.append(([[50 + i, 60, 70 + i, 120], [200, 60, 220, 120], [150, 100, 156, 106], [10, 10, 14, 14]],
                  [0.9, 0.85, 0.6, 0.07], [2, 3, 0, 0]))
    return s


def test_export_writes_valid_schema_and_ranked_ball_alternatives(tmp_path):
    cm = _single_model_map()
    cfg = ExportConfig("m1", "m1-cam", "run1", 25.0)
    meta = export_detections(None, tmp_path, cfg, FakePredictor(RF_NAMES, _script(10)), cm, frames=_frames(10))
    t, b = read_tracks(tmp_path / "tracks.csv"), read_balls(tmp_path / "balls.csv")
    assert validate_tracks(t, "m1", 25.0) == [] and validate_balls(b, "m1", 25.0) == []
    assert set(t.class_name) == {"player", "referee"}
    assert meta["distinct_track_ids"] == 2
    assert len(b) == 20 and set(b.rank_in_frame) == {1, 2}       # low-confidence alternative kept, ranked 2
    assert (b[b.rank_in_frame == 1].confidence == 0.6).all()
    assert (t.foot_y == t.y2).all()


def test_export_empty_video_gives_schema_correct_empty_csvs(tmp_path):
    cm = _single_model_map()
    export_detections(None, tmp_path, ExportConfig("m1", "v", "r", 25.0), FakePredictor(RF_NAMES, []), cm, frames=_frames(5))
    t = pd.read_csv(tmp_path / "tracks.csv")
    assert t.empty and "foot_x" in t.columns


def test_class_map_mismatch_aborts(tmp_path):
    cm = ClassMapConfig.load(REPO / "config/classmaps/roboflow_football.yaml")
    wrong = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle"}
    with pytest.raises(ClassMapError, match="does not match"):
        export_detections(None, tmp_path, ExportConfig("m1", "v", "r", 25.0), FakePredictor(wrong, []), cm,
                          ball_predictor=FakePredictor({0: "ball"}, []), frames=_frames(1))
