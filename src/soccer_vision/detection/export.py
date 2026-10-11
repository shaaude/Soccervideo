"""Step 6: export player/goalkeeper/referee tracks and ball observations to CSV.

Outputs (schemas in soccer_vision.schemas):
  tracks.csv   one row per tracked person per frame (canonical class IDs)
  balls.csv    every ball candidate above the keep threshold, ranked per frame
  detection_meta.json   weights checksums, model class names, thresholds, versions, counts

Track IDs are scoped to (video_id, run_id, segment_id). They are not roster identities.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np

from ..schemas import (BALL_COLUMNS, CLASS_BALL, PERSON_CLASSES, TRACK_COLUMNS,
                       CANONICAL_CLASSES, validate_balls, validate_tracks, read_balls, read_tracks)
from ..provenance import sha256_file
from .classmap import ClassMapConfig, ModelClassMap

CLASS_NAMES = {v: k for k, v in CANONICAL_CLASSES.items()}


@dataclass
class RawDetections:
    xyxy: np.ndarray        # (N,4) float
    confidence: np.ndarray  # (N,)
    class_id: np.ndarray    # (N,) model's own class ids

    @classmethod
    def empty(cls):
        return cls(np.zeros((0, 4), float), np.zeros(0, float), np.zeros(0, int))


class Predictor(Protocol):
    names: dict

    def __call__(self, frame: np.ndarray) -> RawDetections: ...


class UltralyticsPredictor:
    """Thin wrapper so the exporter can be tested without weights (inject any Predictor)."""

    def __init__(self, weights: Path, device: str, imgsz: int, conf: float):
        from ultralytics import YOLO  # AGPL-3.0; imported lazily
        self.model = YOLO(str(weights))
        self.names = dict(self.model.names)
        self.device, self.imgsz, self.conf = device, imgsz, conf

    def __call__(self, frame):
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return RawDetections.empty()
        return RawDetections(b.xyxy.cpu().numpy().astype(float), b.conf.cpu().numpy().astype(float),
                             b.cls.cpu().numpy().astype(int))


@dataclass
class ExportConfig:
    match_id: str
    video_id: str
    run_id: str
    fps: float
    person_conf: float = 0.25
    ball_conf_keep: float = 0.05   # keep low-confidence alternatives for debugging (Step 6)
    max_frames: int | None = None
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8


def _map(dets: RawDetections, cmap: ModelClassMap):
    canon = np.array([cmap.canonical(c) if cmap.canonical(c) is not None else -1 for c in dets.class_id], int)
    keep = canon >= 0
    return dets.xyxy[keep], dets.confidence[keep], canon[keep]


def iter_frames(video_path: Path, max_frames: int | None):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    try:
        i = 0
        while max_frames is None or i < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1
    finally:
        cap.release()


def container_fps(video_path: Path) -> float:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps)


def export_detections(
    video: Path, out_dir: Path, cfg: ExportConfig, person_predictor: Predictor, classmap: ClassMapConfig,
    ball_predictor: Predictor | None = None, frames: Iterable | None = None,
    annotated_video: Path | None = None, weights_info: dict | None = None,
    progress: Callable[[int], None] | None = None,
) -> dict:
    import supervision as sv

    if cfg.fps <= 0:
        raise ValueError("fps must be positive")
    classmap.person_model.check_model_names(person_predictor.names)
    if ball_predictor is not None:
        if classmap.ball_model is None:
            raise ValueError("A ball predictor was given but the class map has no ball_model section")
        classmap.ball_model.check_model_names(ball_predictor.names)
    elif CLASS_BALL not in {CANONICAL_CLASSES[v] for v in classmap.person_model.map.values()}:
        raise ValueError("No ball model and the person model's class map has no 'ball' class")

    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = sv.ByteTrack(track_activation_threshold=cfg.track_activation_threshold,
                           lost_track_buffer=cfg.lost_track_buffer,
                           minimum_matching_threshold=cfg.minimum_matching_threshold,
                           frame_rate=int(round(cfg.fps)))
    frames = frames if frames is not None else iter_frames(video, cfg.max_frames)
    tracks_tmp, balls_tmp = out_dir / "tracks.csv.tmp", out_dir / "balls.csv.tmp"
    sink = None
    n_frames = n_tracks = n_balls = frames_with_ball = 0
    track_ids: set[int] = set()
    t0 = time.time()
    with open(tracks_tmp, "w", newline="") as tf, open(balls_tmp, "w", newline="") as bf:
        tw, bw = csv.writer(tf), csv.writer(bf)
        tw.writerow(TRACK_COLUMNS)
        bw.writerow(BALL_COLUMNS)
        for frame_index, frame in frames:
            t = round(frame_index / cfg.fps, 6)
            raw = person_predictor(frame)
            xyxy, conf, canon = _map(raw, classmap.person_model)
            pmask = np.isin(canon, PERSON_CLASSES) & (conf >= cfg.person_conf)
            people = sv.Detections(xyxy=xyxy[pmask].reshape(-1, 4), confidence=conf[pmask],
                                   class_id=canon[pmask])
            tracked = tracker.update_with_detections(people)
            order = np.argsort(tracked.tracker_id) if len(tracked) else []
            for k in order:
                # Round once, then derive the foot point from the rounded box so they stay consistent.
                x1, y1, x2, y2 = (round(float(v), 2) for v in tracked.xyxy[k])
                tid, cid = int(tracked.tracker_id[k]), int(tracked.class_id[k])
                tw.writerow([cfg.match_id, frame_index, t, tid, cid, round(float(tracked.confidence[k]), 5),
                             x1, y1, x2, y2, round((x1 + x2) / 2, 3), y2,
                             cfg.video_id, cfg.run_id, 0, CLASS_NAMES[cid]])
                n_tracks += 1
                track_ids.add(tid)

            if ball_predictor is not None:
                bxy, bconf, bcanon = _map(ball_predictor(frame), classmap.ball_model)
            else:
                bxy, bconf, bcanon = xyxy, conf, canon
            bmask = (bcanon == CLASS_BALL) & (bconf >= cfg.ball_conf_keep)
            bxy, bconf = bxy[bmask], bconf[bmask]
            if len(bconf):
                frames_with_ball += 1
            for rank, k in enumerate(np.argsort(-bconf), start=1):
                x1, y1, x2, y2 = (round(float(v), 2) for v in bxy[k])
                bw.writerow([cfg.match_id, frame_index, t, round(float(bconf[k]), 5),
                             x1, y1, x2, y2, round((x1 + x2) / 2, 3), round((y1 + y2) / 2, 3),
                             cfg.video_id, cfg.run_id, rank])
                n_balls += 1

            if annotated_video is not None:
                if sink is None:
                    h, w = frame.shape[:2]
                    info = sv.VideoInfo(width=w, height=h, fps=int(round(cfg.fps)))
                    sink = sv.VideoSink(str(annotated_video), info)
                    sink.__enter__()
                    box_ann = sv.EllipseAnnotator(color_lookup=sv.ColorLookup.TRACK)
                    lbl_ann = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_CENTER, text_scale=0.4,
                                                  color_lookup=sv.ColorLookup.TRACK)
                    tri_ann = sv.TriangleAnnotator(base=16, height=14, color=sv.Color.from_hex("#FFD700"))
                scene = box_ann.annotate(frame.copy(), tracked)
                if len(tracked):
                    labels = [f"#{int(i)}" for i in tracked.tracker_id]
                    scene = lbl_ann.annotate(scene, tracked, labels=labels)
                top = (sv.Detections(xyxy=bxy[np.argsort(-bconf)[:1]].reshape(-1, 4), class_id=np.zeros(1, int))
                       if len(bconf) else None)
                if top is not None:
                    scene = tri_ann.annotate(scene, top)
                sink.write_frame(scene)
            n_frames += 1
            if progress:
                progress(frame_index)
    if sink is not None:
        sink.__exit__(None, None, None)

    tracks_path, balls_path = out_dir / "tracks.csv", out_dir / "balls.csv"
    errors = validate_tracks(read_tracks(tracks_tmp), cfg.match_id, cfg.fps) + \
        validate_balls(read_balls(balls_tmp), cfg.match_id, cfg.fps)
    if errors:
        raise ValueError("Exported detections failed schema validation: " + "; ".join(errors))
    tracks_tmp.replace(tracks_path)
    balls_tmp.replace(balls_path)
    meta = {
        "config": asdict(cfg),
        "class_map": classmap.name,
        "person_model_names": {int(k): v for k, v in person_predictor.names.items()},
        "ball_model_names": ({int(k): v for k, v in ball_predictor.names.items()} if ball_predictor else None),
        "weights": weights_info or {},
        "frames_processed": n_frames,
        "track_rows": n_tracks,
        "distinct_track_ids": len(track_ids),
        "ball_rows": n_balls,
        "frames_with_ball_candidate": frames_with_ball,
        "elapsed_seconds": round(time.time() - t0, 2),
        "versions": _versions(),
        "note": "track_id is scoped to (video_id, run_id, segment_id); it is not a roster identity.",
    }
    (out_dir / "detection_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _versions():
    out = {}
    for mod in ("supervision", "ultralytics", "torch", "cv2", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def weights_info(*paths) -> dict:
    return {str(p): sha256_file(p) for p in paths if p}
