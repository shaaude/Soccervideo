"""Step 7: image-to-pitch homographies from surveyed landmark correspondences.

Landmark file (JSON):
{
  "camera_id": "cam01", "field_length_m": 73.0, "field_width_m": 45.0,
  "segments": [
    {"start_frame": 0, "end_frame": 22499,
     "points": [{"name": "corner_near_left", "image_xy": [220, 450], "field_xy": [0, 0]},
                {"name": "halfway_far", "image_xy": [640, 210], "field_xy": [36.5, 45], "holdout": true}, ...]}
  ]
}
Field coordinates: x along the length (0..L), y across the width (0..W), metres.
A segment covers frames where the camera did not pan/tilt/zoom; re-survey after any change.
Frames outside every accepted segment are UNCALIBRATED and get no field coordinates.

Each segment may set "model":
  "homography" (default)  planar camera; >= 4 landmarks
  "poly2" / "poly3"       smooth polynomial image->pitch mapping for stitched/curved images such as a Veo
                          180-degree panorama, where straight pitch lines are curved and one homography
                          cannot fit; >= 8 / >= 14 landmarks. Same held-out error gate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

MIN_POINTS = 4
MODEL_MIN_POINTS = {"homography": 4, "poly2": 8, "poly3": 14}


def _poly_terms(x, y, degree):
    return np.stack([x ** i * y ** (d - i) for d in range(degree + 1) for i in range(d + 1)], axis=1)


def fit_mapping(model: str, img: np.ndarray, fld: np.ndarray, ransac_m: float = 1.0):
    """Return (mapping dict or None, inlier mask)."""
    if model == "homography":
        H, inl = _fit(img, fld, ransac_m)
        return ({"type": "homography", "H": H.tolist()} if H is not None else None), inl
    if model in ("poly2", "poly3"):
        deg = int(model[-1])
        mu, sd = img.mean(0), img.std(0) + 1e-9
        n = (img - mu) / sd
        A = _poly_terms(n[:, 0], n[:, 1], deg)
        cx, *_ = np.linalg.lstsq(A, fld[:, 0], rcond=None)
        cy, *_ = np.linalg.lstsq(A, fld[:, 1], rcond=None)
        return ({"type": model, "degree": deg, "mu": mu.tolist(), "sd": sd.tolist(),
                 "coef_x": cx.tolist(), "coef_y": cy.tolist()}, np.ones(len(img), bool))
    raise ValueError(f"unknown calibration model '{model}'")


def apply_mapping(mapping: dict, pts) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 2)
    if mapping["type"] == "homography":
        return project(np.array(mapping["H"]), pts)
    n = (pts - np.array(mapping["mu"])) / np.array(mapping["sd"])
    A = _poly_terms(n[:, 0], n[:, 1], mapping["degree"])
    return np.stack([A @ np.array(mapping["coef_x"]), A @ np.array(mapping["coef_y"])], axis=1)


def standard_landmarks(length_m: float, width_m: float) -> dict[str, tuple[float, float]]:
    """Format-independent landmarks (any 4v4..11v11 pitch with verified length/width)."""
    L, W = float(length_m), float(width_m)
    return {
        "corner_near_left": (0.0, 0.0), "corner_far_left": (0.0, W),
        "corner_near_right": (L, 0.0), "corner_far_right": (L, W),
        "halfway_near": (L / 2, 0.0), "halfway_far": (L / 2, W), "centre_spot": (L / 2, W / 2),
    }


@dataclass
class SegmentCalibration:
    start_frame: int
    end_frame: int
    H: np.ndarray | None
    n_fit_points: int
    n_inliers: int
    fit_rmse_m: float | None
    holdout_errors_m: list[float] = field(default_factory=list)
    loo_errors_m: list[float] = field(default_factory=list)
    status: str = "uncalibrated"
    reason: str = ""
    model: str = "homography"
    mapping: dict | None = None

    @property
    def check_errors_m(self) -> list[float]:
        return self.holdout_errors_m or self.loo_errors_m

    def to_dict(self):
        e = self.check_errors_m
        return {
            "start_frame": self.start_frame, "end_frame": self.end_frame,
            "model": self.model,
            "H": self.H.tolist() if self.H is not None else None,
            "mapping": self.mapping,
            "n_fit_points": self.n_fit_points, "n_inliers": self.n_inliers, "fit_rmse_m": self.fit_rmse_m,
            "holdout_errors_m": self.holdout_errors_m, "leave_one_out_errors_m": self.loo_errors_m,
            "check_method": "holdout" if self.holdout_errors_m else ("leave_one_out" if self.loo_errors_m else "none"),
            "median_check_error_m": float(np.median(e)) if e else None,
            "status": self.status, "reason": self.reason,
        }


def _fit(img: np.ndarray, fld: np.ndarray, ransac_px: float):
    import cv2
    H, mask = cv2.findHomography(img.astype(np.float32), fld.astype(np.float32), cv2.RANSAC, ransac_px)
    return H, (mask.ravel().astype(bool) if mask is not None else np.zeros(len(img), bool))


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    import cv2
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _collinear(pts: np.ndarray, tol=1e-3) -> bool:
    c = pts - pts.mean(axis=0)
    s = np.linalg.svd(c, compute_uv=False)
    return s[-1] / (s[0] + 1e-12) < tol


def calibrate_segment(seg: dict, length_m: float, width_m: float, ransac_m: float = 1.0,
                      max_median_error_m: float = 2.0) -> SegmentCalibration:
    pts = seg["points"]
    img = np.array([p["image_xy"] for p in pts], float)
    fld = np.array([p["field_xy"] for p in pts], float)
    hold = np.array([bool(p.get("holdout", False)) for p in pts])
    model = seg.get("model", "homography")
    sc = SegmentCalibration(int(seg["start_frame"]), int(seg["end_frame"]), None, int((~hold).sum()), 0, None,
                            model=model)
    if model not in MODEL_MIN_POINTS:
        sc.reason = f"unknown model '{model}'"
        return sc
    need = MODEL_MIN_POINTS[model]
    tol = 0.5
    if ((fld[:, 0] < -tol) | (fld[:, 0] > length_m + tol) | (fld[:, 1] < -tol) | (fld[:, 1] > width_m + tol)).any():
        sc.reason = "field coordinates outside the verified pitch dimensions"
        return sc
    fit_img, fit_fld = img[~hold], fld[~hold]
    if len(fit_img) < need:
        sc.reason = f"need >= {need} non-holdout landmarks for {model}, have {len(fit_img)}"
        return sc
    if _collinear(fit_img) or _collinear(fit_fld):
        sc.reason = "landmarks are collinear"
        return sc
    # RANSAC threshold (homography) is in destination units (metres).
    mapping, inl = fit_mapping(model, fit_img, fit_fld, ransac_m)
    if mapping is None:
        sc.reason = f"{model} estimation failed"
        return sc
    sc.mapping, sc.n_inliers = mapping, int(inl.sum())
    sc.H = np.array(mapping["H"]) if model == "homography" else None
    resid = np.linalg.norm(apply_mapping(mapping, fit_img) - fit_fld, axis=1)
    sc.fit_rmse_m = float(np.sqrt(np.mean(resid[inl] ** 2))) if inl.any() else None
    if hold.any():
        sc.holdout_errors_m = [float(x) for x in np.linalg.norm(apply_mapping(mapping, img[hold]) - fld[hold], axis=1)]
    elif len(fit_img) > need:
        for i in range(len(fit_img)):
            m = np.ones(len(fit_img), bool)
            m[i] = False
            if _collinear(fit_img[m]):
                continue
            mi, _ = fit_mapping(model, fit_img[m], fit_fld[m], ransac_m)
            if mi is not None:
                sc.loo_errors_m.append(float(np.linalg.norm(apply_mapping(mi, fit_img[i:i + 1])[0] - fit_fld[i])))
    errs = sc.check_errors_m
    if not errs:
        sc.status, sc.reason = "calibrated_unvalidated", "no holdout or leave-one-out check possible (add landmarks)"
    elif float(np.median(errs)) <= max_median_error_m:
        sc.status = "calibrated"
    else:
        sc.status, sc.reason = "rejected", f"median check error {np.median(errs):.2f} m > {max_median_error_m} m"
        sc.H, sc.mapping = None, None
    return sc


def calibrate(landmarks_path, max_median_error_m: float = 2.0, accept_unvalidated: bool = False) -> dict:
    spec = json.loads(Path(landmarks_path).read_text())
    L, W = spec.get("field_length_m"), spec.get("field_width_m")
    if not L or not W:
        raise ValueError("Landmark file must state verified field_length_m and field_width_m")
    segs = [calibrate_segment(s, L, W, max_median_error_m=max_median_error_m) for s in spec["segments"]]
    spans = sorted((s.start_frame, s.end_frame) for s in segs)
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        if b0 <= a1:
            raise ValueError(f"Calibration segments overlap: {a0}-{a1} and {b0}-{b1}")
    if not accept_unvalidated:
        for s in segs:
            if s.status == "calibrated_unvalidated":
                s.status, s.H, s.mapping = "rejected", None, None
                s.reason += "; unvalidated transforms not accepted"
    return {"camera_id": spec.get("camera_id"), "field_length_m": L, "field_width_m": W,
            "max_median_error_m": max_median_error_m,
            "segments": [s.to_dict() for s in segs]}


def project_tracks(tracks: pd.DataFrame, transforms: dict) -> pd.DataFrame:
    """Add field_x_m/field_y_m from foot points. Uncalibrated frames stay NaN (missing != zero).

    Ball centres are deliberately not projected: an airborne ball is not on the ground plane.
    """
    out = tracks.copy()
    out["field_x_m"] = np.nan
    out["field_y_m"] = np.nan
    out["calibrated"] = False
    out["calibration_segment"] = -1
    L, W = transforms["field_length_m"], transforms["field_width_m"]
    for i, seg in enumerate(transforms["segments"]):
        mapping = seg.get("mapping") or ({"type": "homography", "H": seg["H"]} if seg.get("H") else None)
        if mapping is None or seg["status"] not in ("calibrated", "calibrated_unvalidated"):
            continue
        m = out["frame_index"].between(seg["start_frame"], seg["end_frame"])
        if not m.any():
            continue
        xy = apply_mapping(mapping, out.loc[m, ["foot_x", "foot_y"]].to_numpy(float))
        out.loc[m, "field_x_m"], out.loc[m, "field_y_m"] = xy[:, 0], xy[:, 1]
        out.loc[m, "calibrated"] = True
        out.loc[m, "calibration_segment"] = i
    # Off-pitch projections (beyond a 5 m margin) are implausible for a ground contact: flag them.
    off = out["calibrated"] & ((out["field_x_m"] < -5) | (out["field_x_m"] > L + 5) |
                               (out["field_y_m"] < -5) | (out["field_y_m"] > W + 5))
    out["off_pitch"] = off
    return out
