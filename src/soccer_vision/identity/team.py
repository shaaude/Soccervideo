"""Step 8.2: unsupervised team-appearance clustering per tracklet.

Output cluster indices (0/1) are NOT teams. A reviewer must map clusters to the verified
home/away team IDs (team_cluster_mapping.json) before any team attribution is used.
Referees and goalkeepers are excluded from clustering (different kits).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..schemas import CLASS_PLAYER


def read_frames(video, wanted: set[int]):
    """Sequential decode (seeking per frame in H.264 is far slower); yields only wanted frames."""
    import cv2
    if not wanted:
        return
    last = max(wanted)
    cap = cv2.VideoCapture(str(video))
    try:
        f = 0
        while f <= last:
            ok = cap.grab()
            if not ok:
                break
            if f in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    yield f, frame
            f += 1
    finally:
        cap.release()


def torso_crop(frame: np.ndarray, box) -> np.ndarray | None:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = y2 - y1, x2 - x1
    if h < 12 or w < 6:
        return None
    ty1, ty2 = y1 + int(0.15 * h), y1 + int(0.55 * h)
    tx1, tx2 = x1 + int(0.2 * w), x2 - int(0.2 * w)
    H, W = frame.shape[:2]
    ty1, ty2, tx1, tx2 = max(0, ty1), min(H, ty2), max(0, tx1), min(W, tx2)
    if ty2 - ty1 < 4 or tx2 - tx1 < 3:
        return None
    return frame[ty1:ty2, tx1:tx2]


def color_feature(crop_bgr: np.ndarray) -> np.ndarray:
    import cv2
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    # Drop the greenest pixels (grass bleeding into the crop): lowest a* channel quartile.
    keep = lab[:, 1] > np.quantile(lab[:, 1], 0.25)
    px = lab[keep] if keep.sum() >= 10 else lab
    return np.concatenate([np.median(px, axis=0), px.std(axis=0)])


def kmeans(X: np.ndarray, k: int = 2, seed: int = 42, iters: int = 100):
    rng = np.random.default_rng(seed)
    centers = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
        centers.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
    C = np.array(centers)
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    D = np.sqrt(((X[:, None, :] - C[None]) ** 2).sum(-1))
    return lab, C, D


def cluster_teams(video, tracks: pd.DataFrame, samples_per_track: int = 8, min_samples: int = 3,
                  seed: int = 42) -> pd.DataFrame:
    players = tracks[tracks["class_id"] == CLASS_PLAYER]
    want: dict[int, list] = {}
    for tid, g in players.groupby("track_id"):
        g = g.sort_values("frame_index")
        idx = np.linspace(0, len(g) - 1, min(samples_per_track, len(g))).round().astype(int)
        for r in g.iloc[np.unique(idx)].itertuples(index=False):
            want.setdefault(int(r.frame_index), []).append((int(tid), (r.x1, r.y1, r.x2, r.y2)))
    feats: dict[int, list] = {}
    for f, frame in read_frames(video, set(want)):
        for tid, box in want[f]:
            crop = torso_crop(frame, box)
            if crop is not None:
                feats.setdefault(tid, []).append(color_feature(crop))
    rows = [(tid, len(v), np.median(np.array(v), axis=0)) for tid, v in feats.items() if len(v) >= min_samples]
    cols = ["track_id", "n_samples", "cluster", "distance_margin", "team_id", "mapping_status"]
    if len(rows) < 2:
        return pd.DataFrame(columns=cols)
    X = np.array([r[2] for r in rows])
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)
    lab, _, D = kmeans(Xn, 2, seed)
    Ds = np.sort(D, axis=1)
    margin = (Ds[:, 1] - Ds[:, 0]) / (Ds[:, 1] + 1e-9)
    return pd.DataFrame({"track_id": [r[0] for r in rows], "n_samples": [r[1] for r in rows],
                         "cluster": lab.astype(int), "distance_margin": np.round(margin, 3),
                         "team_id": "", "mapping_status": "unreviewed"}, columns=cols)
