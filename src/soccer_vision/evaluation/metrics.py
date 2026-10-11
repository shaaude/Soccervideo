"""Step 18: independent quality evaluation. Every metric reports its numerator and denominator.

Ground truth must be independently annotated (including events the model missed) and split by
whole matches. Nothing here produces a number without labelled data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ratio(n, d):
    return round(n / d, 4) if d else None


def match_events(pred: pd.DataFrame, truth: pd.DataFrame, tolerance_s: float = 1.0,
                 type_col: str = "event_type", time_col: str = "video_time_s") -> pd.DataFrame:
    """Greedy one-to-one matching per (match_id, event_type) by closest time within tolerance."""
    pairs = []
    for (mid, et), t in truth.groupby(["match_id", type_col]):
        p = pred[(pred["match_id"] == mid) & (pred[type_col] == et)]
        cand = []
        for ti, tr in t.iterrows():
            for pi, pr in p.iterrows():
                dt = abs(float(pr[time_col]) - float(tr[time_col]))
                if dt <= tolerance_s:
                    cand.append((dt, ti, pi))
        used_t, used_p = set(), set()
        for dt, ti, pi in sorted(cand):
            if ti in used_t or pi in used_p:
                continue
            used_t.add(ti)
            used_p.add(pi)
            pairs.append({"match_id": mid, "event_type": et, "truth_idx": ti, "pred_idx": pi, "abs_time_error_s": dt})
    return pd.DataFrame(pairs, columns=["match_id", "event_type", "truth_idx", "pred_idx", "abs_time_error_s"])


def event_metrics(pred: pd.DataFrame, truth: pd.DataFrame, tolerance_s: float = 1.0,
                  actor_col: str = "actor_player_id") -> pd.DataFrame:
    """Per-class precision/recall/F1, timing error, and joint event+actor accuracy (Step 11.8)."""
    pairs = match_events(pred, truth, tolerance_s)
    rows = []
    classes = sorted(set(truth["event_type"]).union(pred["event_type"]))
    for et in classes + ["__all__"]:
        sel = pairs if et == "__all__" else pairs[pairs["event_type"] == et]
        n_t = len(truth) if et == "__all__" else int((truth["event_type"] == et).sum())
        n_p = len(pred) if et == "__all__" else int((pred["event_type"] == et).sum())
        tp = len(sel)
        prec, rec = _ratio(tp, n_p), _ratio(tp, n_t)
        f1 = round(2 * prec * rec / (prec + rec), 4) if prec and rec else (0.0 if n_p and n_t else None)
        joint = None
        if actor_col in pred and actor_col in truth and tp:
            same = sum(str(pred.loc[r.pred_idx, actor_col]) == str(truth.loc[r.truth_idx, actor_col])
                       and str(truth.loc[r.truth_idx, actor_col]).strip() != "" for r in sel.itertuples())
            joint_tp = same
            jp, jr = _ratio(joint_tp, n_p), _ratio(joint_tp, n_t)
            joint = round(2 * jp * jr / (jp + jr), 4) if jp and jr else 0.0
        rows.append({"event_type": et, "true_positives": tp, "predicted": n_p, "ground_truth": n_t,
                     "false_positives": n_p - tp, "false_negatives": n_t - tp,
                     "precision": prec, "recall": rec, "f1": f1,
                     "median_abs_time_error_s": round(float(sel["abs_time_error_s"].median()), 3) if tp else None,
                     "joint_event_actor_f1": joint, "tolerance_s": tolerance_s})
    return pd.DataFrame(rows)


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def detection_recall(pred: pd.DataFrame, truth: pd.DataFrame, iou_threshold: float = 0.5,
                     class_col: str = "class_id") -> pd.DataFrame:
    """Recall of ground-truth boxes per class (players: all GT boxes; ball: only frames where ball is visible)."""
    rows = []
    for cls, t in truth.groupby(class_col):
        hit = 0
        for f, tf in t.groupby("frame_index"):
            pf = pred[(pred["frame_index"] == f) & (pred[class_col] == cls)]
            boxes = pf[["x1", "y1", "x2", "y2"]].to_numpy(float)
            used = set()
            for tb in tf[["x1", "y1", "x2", "y2"]].to_numpy(float):
                best, bi = 0.0, None
                for i, pb in enumerate(boxes):
                    if i in used:
                        continue
                    v = iou(tb, pb)
                    if v > best:
                        best, bi = v, i
                if bi is not None and best >= iou_threshold:
                    used.add(bi)
                    hit += 1
        rows.append({class_col: cls, "matched": hit, "ground_truth_boxes": len(t), "recall": _ratio(hit, len(t)),
                     "iou_threshold": iou_threshold})
    return pd.DataFrame(rows)


def identity_metrics(assigned: pd.DataFrame, truth: pd.DataFrame, key=("video_id", "track_id"),
                     high_conf: float = 0.9) -> dict:
    """Correct / unknown / incorrect attribution, and wrong high-confidence identity rate (Step 25)."""
    m = truth.merge(assigned, on=list(key), how="left", suffixes=("_true", "_pred"))
    pred = m["player_id_pred"].fillna("").astype(str).str.strip()
    true = m["player_id_true"].astype(str)
    unknown = (pred == "") | (pred.str.upper() == "UNKNOWN")
    correct = ~unknown & (pred == true)
    wrong = ~unknown & (pred != true)
    conf = pd.to_numeric(m.get("identity_confidence", pd.Series(np.nan, index=m.index)), errors="coerce")
    hc = ~unknown & (conf >= high_conf)
    return {"tracklets": len(m), "correct": int(correct.sum()), "unknown": int(unknown.sum()),
            "incorrect": int(wrong.sum()), "accuracy_of_assigned": _ratio(int(correct.sum()), int((~unknown).sum())),
            "abstention_rate": _ratio(int(unknown.sum()), len(m)),
            "high_conf_assigned": int(hc.sum()), "high_conf_wrong": int((hc & wrong).sum()),
            "wrong_high_conf_rate": _ratio(int((hc & wrong).sum()), int(hc.sum()))}


def tracking_metrics(pred: pd.DataFrame, truth: pd.DataFrame, iou_threshold: float = 0.5) -> dict:
    """IDF1 / MOTA / ID switches via py-motmetrics (pip install motmetrics)."""
    import motmetrics as mm
    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(truth["frame_index"]).union(pred["frame_index"]))
    for f in frames:
        t = truth[truth["frame_index"] == f]
        p = pred[pred["frame_index"] == f]
        tb = t[["x1", "y1", "x2", "y2"]].to_numpy(float)
        pb = p[["x1", "y1", "x2", "y2"]].to_numpy(float)
        tb[:, 2:] -= tb[:, :2]
        pb[:, 2:] -= pb[:, :2]
        d = mm.distances.iou_matrix(tb, pb, max_iou=1 - iou_threshold)
        acc.update(t["track_id"].tolist(), p["track_id"].tolist(), d)
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["idf1", "mota", "num_switches", "num_false_positives", "num_misses",
                                 "num_objects"], name="all")
    return {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in s.iloc[0].items()}
