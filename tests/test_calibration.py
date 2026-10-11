import json

import numpy as np
import pandas as pd
import pytest

from soccer_vision.calibration.homography import calibrate, project, project_tracks, standard_landmarks

L, W = 70.0, 45.0
# Ground-truth homography field(m) -> image(px) for a plausible elevated camera
H_true_f2i = np.array([[12.0, 3.0, 200.0], [0.5, -6.0, 520.0], [0.0004, 0.004, 1.0]])


def to_img(fxy):
    return project(H_true_f2i, np.array(fxy, float))


def landmark_file(tmp_path, holdout=True, noise=0.0, outlier=False, extra=True):
    names = standard_landmarks(L, W)
    pts = dict(names)
    if extra:
        pts.update({"pa_near_left": (16.5, 12.0), "pa_far_left": (16.5, 33.0), "pa_near_right": (53.5, 12.0)})
    rng = np.random.default_rng(0)
    items = []
    for n, f in pts.items():
        img = to_img([f])[0] + rng.normal(0, noise, 2)
        items.append({"name": n, "image_xy": img.tolist(), "field_xy": list(f),
                      "holdout": holdout and n in ("centre_spot", "pa_near_right")})
    if outlier:
        items[1]["image_xy"] = [items[1]["image_xy"][0] + 300, items[1]["image_xy"][1]]
    spec = {"camera_id": "cam01", "field_length_m": L, "field_width_m": W,
            "segments": [{"start_frame": 0, "end_frame": 99, "points": items}]}
    p = tmp_path / "lm.json"
    p.write_text(json.dumps(spec))
    return p


def test_recovers_known_geometry_with_holdout_check(tmp_path):
    t = calibrate(landmark_file(tmp_path))
    s = t["segments"][0]
    assert s["status"] == "calibrated" and s["check_method"] == "holdout"
    assert s["median_check_error_m"] < 0.01


def test_ransac_rejects_single_outlier(tmp_path):
    t = calibrate(landmark_file(tmp_path, outlier=True))
    s = t["segments"][0]
    assert s["n_inliers"] == s["n_fit_points"] - 1
    assert s["median_check_error_m"] < 0.05


def test_leave_one_out_used_without_holdouts(tmp_path):
    s = calibrate(landmark_file(tmp_path, holdout=False))["segments"][0]
    assert s["check_method"] == "leave_one_out" and s["status"] == "calibrated"


def test_large_error_rejects_segment(tmp_path):
    s = calibrate(landmark_file(tmp_path, noise=40.0), max_median_error_m=0.5)["segments"][0]
    assert s["status"] == "rejected" and s["H"] is None


def test_too_few_points(tmp_path):
    p = landmark_file(tmp_path, extra=False)
    spec = json.loads(p.read_text())
    spec["segments"][0]["points"] = spec["segments"][0]["points"][:3]
    p.write_text(json.dumps(spec))
    s = calibrate(p)["segments"][0]
    assert s["status"] == "uncalibrated" and "need >= 4" in s["reason"]


def test_projection_only_inside_calibrated_frames(tmp_path):
    t = calibrate(landmark_file(tmp_path))
    foot = to_img([[30.0, 20.0]])[0]
    tracks = pd.DataFrame({"frame_index": [10, 150], "foot_x": [foot[0]] * 2, "foot_y": [foot[1]] * 2})
    out = project_tracks(tracks, t)
    assert out.loc[0, "calibrated"] and abs(out.loc[0, "field_x_m"] - 30) < 0.05 and abs(out.loc[0, "field_y_m"] - 20) < 0.05
    assert not out.loc[1, "calibrated"] and np.isnan(out.loc[1, "field_x_m"])  # missing, not zero


def test_overlapping_segments_rejected(tmp_path):
    p = landmark_file(tmp_path)
    spec = json.loads(p.read_text())
    spec["segments"].append({**spec["segments"][0], "start_frame": 50, "end_frame": 120})
    p.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="overlap"):
        calibrate(p)


def _panorama_like(fxy):
    """Homography followed by a cylindrical horizontal warp, as in a stitched 180-degree panorama."""
    p = to_img(fxy)
    cx, f = 600.0, 300.0   # strong curvature, as across a 180-degree stitched view
    return np.stack([cx + f * np.arctan((p[:, 0] - cx) / f), p[:, 1] + 0.00025 * (p[:, 0] - cx) ** 2], axis=1)


def _grid_landmarks(tmp_path, model):
    xs, ys = np.linspace(0, L, 6), np.linspace(0, W, 4)
    pts = [(x, y) for x in xs for y in ys]
    img = _panorama_like(pts)
    hold = {3, 10, 17}
    items = [{"image_xy": img[i].tolist(), "field_xy": list(pts[i]), "holdout": i in hold} for i in range(len(pts))]
    spec = {"field_length_m": L, "field_width_m": W,
            "segments": [{"start_frame": 0, "end_frame": 10, "model": model, "points": items}]}
    p = tmp_path / f"pano-{model}.json"
    p.write_text(json.dumps(spec))
    return p


def test_curved_panorama_rejects_homography_but_poly3_passes(tmp_path):
    h = calibrate(_grid_landmarks(tmp_path, "homography"))["segments"][0]
    p3 = calibrate(_grid_landmarks(tmp_path, "poly3"))["segments"][0]
    assert h["status"] == "rejected"
    assert p3["status"] == "calibrated" and p3["median_check_error_m"] < h["median_check_error_m"] / 5
    t = calibrate(_grid_landmarks(tmp_path, "poly3"))
    foot = _panorama_like([[30.0, 20.0]])[0]
    out = project_tracks(pd.DataFrame({"frame_index": [5], "foot_x": [foot[0]], "foot_y": [foot[1]]}), t)
    assert abs(out.field_x_m[0] - 30) < 1.0 and abs(out.field_y_m[0] - 20) < 1.0


def test_poly3_needs_enough_landmarks(tmp_path):
    p = landmark_file(tmp_path)
    spec = json.loads(p.read_text())
    spec["segments"][0]["model"] = "poly3"
    p.write_text(json.dumps(spec))
    s = calibrate(p)["segments"][0]
    assert s["status"] == "uncalibrated" and "need >= 14" in s["reason"]
