"""Tiled inference for very wide frames (Veo panorama).

A stitched 180-degree panorama shrinks far-side players to a few pixels when the whole frame is resized to
the detector's input size. Running the detector on overlapping tiles at native resolution keeps them
detectable. Detections from neighbouring tiles are merged by class-wise NMS plus containment suppression
(a box cut by a tile border is usually contained in the full box from the neighbouring tile).
"""
from __future__ import annotations

import numpy as np

from .export import RawDetections


def tile_origins(length: int, tile: int, overlap: float) -> list[int]:
    if length <= tile:
        return [0]
    stride = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, length - tile + 1, stride))
    if xs[-1] != length - tile:
        xs.append(length - tile)
    return xs


def _iou_and_containment(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    iou = inter / np.maximum(a + b - inter, 1e-9)
    contain = inter / np.maximum(np.minimum(a, b), 1e-9)
    return iou, contain


def merge(xyxy, conf, cls, iou_thr=0.5, contain_thr=0.8):
    keep = []
    for c in np.unique(cls):
        idx = np.flatnonzero(cls == c)
        idx = idx[np.argsort(-conf[idx])]
        while len(idx):
            i, rest = idx[0], idx[1:]
            keep.append(i)
            if not len(rest):
                break
            iou, contain = _iou_and_containment(xyxy[i], xyxy[rest])
            idx = rest[(iou < iou_thr) & (contain < contain_thr)]
    keep = np.array(sorted(keep), int)
    return xyxy[keep], conf[keep], cls[keep]


class TiledPredictor:
    def __init__(self, base, tile: int = 1280, overlap: float = 0.2, min_frame_ratio: float = 1.25):
        if not 0 <= overlap < 0.9:
            raise ValueError("overlap must be in [0, 0.9)")
        self.base, self.tile, self.overlap, self.min_ratio = base, int(tile), float(overlap), min_frame_ratio
        self.names = base.names

    def __call__(self, frame: np.ndarray) -> RawDetections:
        h, w = frame.shape[:2]
        if w <= self.tile * self.min_ratio and h <= self.tile * self.min_ratio:
            return self.base(frame)
        boxes, confs, clss = [], [], []
        for y0 in tile_origins(h, self.tile, self.overlap):
            for x0 in tile_origins(w, self.tile, self.overlap):
                d = self.base(frame[y0:y0 + self.tile, x0:x0 + self.tile])
                if len(d.confidence):
                    boxes.append(d.xyxy + np.array([x0, y0, x0, y0], float))
                    confs.append(d.confidence)
                    clss.append(d.class_id)
        if not boxes:
            return RawDetections.empty()
        xyxy, conf, cls = merge(np.vstack(boxes), np.concatenate(confs), np.concatenate(clss))
        return RawDetections(xyxy, conf, cls)
