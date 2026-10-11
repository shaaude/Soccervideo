"""Match report: one self-contained HTML page per processing run (report.html in the run directory).

Everything shown is computed from the run's own files. Unreviewed candidates are labelled as such and
player statistics appear only from human-confirmed events (the four invariants still hold here).
Open it in any browser; it works offline and can be shared as a single file.
"""
from __future__ import annotations

import base64
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import read_review_csv

# Reference palette (dataviz skill): categorical slots 1-2 for the two kit-colour groups,
# sequential blue for density. Light and dark steps are both defined below.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
ORANGE_RAMP = ["#fde0d2", "#f9b897", "#f48f5f", "#eb6834", "#d95926", "#b44619", "#8a3410"]


def _esc(s) -> str:
    return html.escape(str(s))


def _hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


def _ramp_lut(ramp):
    """256-entry RGB lookup from a list of hex stops."""
    stops = np.array([_hex_rgb(c) for c in ramp])
    x = np.linspace(0, 1, len(stops))
    t = np.linspace(0, 1, 256)
    return np.stack([np.interp(t, x, stops[:, k]) for k in range(3)], axis=1)


def _frame(video: Path, t_frac: float = 0.5):
    import cv2
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * t_frac))
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"cannot read a frame from {video}")
    return f


def density_image(frame, xs, ys, ramp, width=1600, sigma_px=14) -> str:
    """Density of foot points over a dimmed, desaturated frame -> base64 JPEG."""
    import cv2
    h, w = frame.shape[:2]
    grey = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR).astype(float)
    base = 0.55 * grey + 0.45 * frame.astype(float)
    base = base * 0.62 + 18
    dens = np.zeros((h, w), float)
    xi = np.clip(np.asarray(xs, int), 0, w - 1)
    yi = np.clip(np.asarray(ys, int), 0, h - 1)
    np.add.at(dens, (yi, xi), 1.0)
    dens = cv2.GaussianBlur(dens, (0, 0), sigma_px)
    if dens.max() > 0:
        dens = np.sqrt(dens / dens.max())          # sqrt: shows the spread, not only the hottest spot
    lut = _ramp_lut(ramp)[:, ::-1]                  # RGB -> BGR
    col = lut[(dens * 255).astype(int)]
    alpha = np.clip(dens * 1.4, 0, 0.85)[..., None]
    out = base * (1 - alpha) + col * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    scale = width / w
    out = cv2.resize(out, (width, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# ----------------------------------------------------------------------------- SVG charts
def _nice_ticks(vmax, n=4):
    if vmax <= 0:
        return [0, 1]
    raw = vmax / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    return list(np.arange(0, vmax + step * 0.999, step))


def line_chart(t, y, title_id, y_label, width=960, height=200, events=None) -> str:
    """Single-series line over time with crosshair tooltip; event ticks along the bottom."""
    pad_l, pad_r, pad_t, pad_b = 44, 12, 12, 30
    tmax = float(max(t)) if len(t) else 1.0
    ymax = float(max(y)) if len(y) else 1.0
    ticks = _nice_ticks(ymax)
    ytop = ticks[-1] or 1
    X = lambda v: pad_l + (width - pad_l - pad_r) * v / tmax
    Y = lambda v: pad_t + (height - pad_t - pad_b) * (1 - v / ytop)
    pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(t, y))
    grid = "".join(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                   f'<text x="{pad_l - 6}" y="{Y(v) + 4:.1f}" class="tick" text-anchor="end">{v:g}</text>' for v in ticks)
    xt = _nice_ticks(tmax, 6)
    xaxis = "".join(f'<text x="{X(v):.1f}" y="{height - 8}" class="tick" text-anchor="middle">{_mmss(v)}</text>'
                    for v in xt if v <= tmax)
    ev = ""
    for e in events or []:
        ev += (f'<line x1="{X(e["t"]):.1f}" x2="{X(e["t"]):.1f}" y1="{pad_t}" y2="{height - pad_b}" class="evline"/>')
    data = json.dumps([[round(float(a), 2), round(float(b), 2)] for a, b in zip(t, y)])
    return f'''<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-labelledby="{title_id}"
  data-series='{data}' data-x0="{pad_l}" data-x1="{width - pad_r}" data-tmax="{tmax}" data-ylabel="{_esc(y_label)}">
  {grid}{xaxis}{ev}
  <polyline points="{pts}" class="series1" fill="none"/>
  <line class="cross" x1="0" x2="0" y1="{pad_t}" y2="{height - pad_b}" visibility="hidden"/>
  <circle class="crossdot" r="4" visibility="hidden"/>
  <rect x="{pad_l}" y="{pad_t}" width="{width - pad_l - pad_r}" height="{height - pad_t - pad_b}" fill="transparent" class="hit"/>
</svg>'''


def strip_chart(t_all, present, events, tmax, width=960, height=64) -> str:
    """Ball-visible strip (1 = seen) with event markers that carry tooltips."""
    pad_l, pad_r = 44, 12
    X = lambda v: pad_l + (width - pad_l - pad_r) * v / max(tmax, 1e-9)
    # merge consecutive seen samples into runs
    runs, start, prev = [], None, None
    dt = (t_all[1] - t_all[0]) if len(t_all) > 1 else 0.1
    for tt, p in zip(t_all, present):
        if p and start is None:
            start = tt
        if not p and start is not None:
            runs.append((start, prev + dt))
            start = None
        prev = tt
    if start is not None:
        runs.append((start, prev + dt))
    bars = "".join(f'<rect x="{X(a):.1f}" y="10" width="{max(1.0, X(b) - X(a)):.1f}" height="16" rx="2" class="seen"/>'
                   for a, b in runs)
    marks = "".join(
        f'<g class="evmark" data-tip="{_esc(e["tip"])}"><circle cx="{X(e["t"]):.1f}" cy="46" r="6" class="ev-{e["status"]}"/>'
        f'<circle cx="{X(e["t"]):.1f}" cy="46" r="13" fill="transparent"/></g>' for e in events)
    return f'''<svg viewBox="0 0 {width} {height}" class="chart strip" role="img" aria-label="Ball visibility and possible passes over time">
  <text x="{pad_l - 6}" y="22" class="tick" text-anchor="end">ball</text>
  <rect x="{pad_l}" y="10" width="{width - pad_l - pad_r}" height="16" rx="2" class="unseen"/>{bars}
  <text x="{pad_l - 6}" y="50" class="tick" text-anchor="end">passes?</text>{marks}
</svg>'''


def histogram(values, bins, width=460, height=200, xlabel="") -> str:
    counts, edges = np.histogram(values, bins=bins)
    pad_l, pad_r, pad_t, pad_b = 40, 8, 10, 34
    ticks = _nice_ticks(counts.max() if len(counts) else 1)
    ytop = ticks[-1] or 1
    bw = (width - pad_l - pad_r) / len(counts)
    Y = lambda v: pad_t + (height - pad_t - pad_b) * (1 - v / ytop)
    grid = "".join(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                   f'<text x="{pad_l - 6}" y="{Y(v) + 4:.1f}" class="tick" text-anchor="end">{v:g}</text>' for v in ticks)
    bars = ""
    for i, c in enumerate(counts):
        x = pad_l + i * bw + 1
        lab = f"{edges[i]:g}–{edges[i + 1]:g} s"
        bars += (f'<g class="bar" data-tip="{_esc(lab)}: {c} tracks"><rect x="{x:.1f}" y="{Y(c):.1f}" width="{bw - 2:.1f}" '
                 f'height="{max(0, Y(0) - Y(c)):.1f}" rx="3" class="series1fill"/>'
                 f'<rect x="{x:.1f}" y="{pad_t}" width="{bw - 2:.1f}" height="{height - pad_t - pad_b}" fill="transparent"/></g>'
                 f'<text x="{x + (bw - 2) / 2:.1f}" y="{height - 18}" class="tick" text-anchor="middle">{edges[i]:g}</text>')
    return f'''<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{_esc(xlabel)}">{grid}{bars}
  <text x="{(width + pad_l) / 2:.0f}" y="{height - 2}" class="tick" text-anchor="middle">{_esc(xlabel)}</text></svg>'''


def pitch_zones_svg(grid, length_m, width_m, width=620) -> str:
    """Top-down pitch with ball-time share per zone (sequential blue), labelled with percentages."""
    ny, nx = grid.shape
    pad = 14
    W = width - 2 * pad
    H = W * width_m / length_m
    sx, sy = W / length_m, H / width_m
    lut = _ramp_lut(BLUE_RAMP)
    vmax = grid.max() or 1
    cells = ""
    for j in range(ny):
        for i in range(nx):
            v = grid[j, i]
            c = lut[int(40 + 215 * v / vmax)] if v > 0 else None
            fill = f"rgb({c[0]:.0f},{c[1]:.0f},{c[2]:.0f})" if c is not None else "transparent"
            x, y = pad + i * W / nx, pad + j * H / ny
            ink = "#ffffff" if v / vmax > 0.55 else "var(--text-primary)"
            cells += (f'<g data-tip="{v * 100:.0f}% of ball time"><rect x="{x + 1:.1f}" y="{y + 1:.1f}" width="{W / nx - 2:.1f}" '
                      f'height="{H / ny - 2:.1f}" rx="3" fill="{fill}"/>'
                      f'<text x="{x + W / nx / 2:.1f}" y="{y + H / ny / 2 + 4:.1f}" text-anchor="middle" class="zlab" fill="{ink}">'
                      f'{v * 100:.0f}%</text></g>')
    cx, cy = pad + W / 2, pad + H / 2
    pa_d, pa_w = 16.5 * sx, 40.3 * sy
    lines = (f'<rect x="{pad}" y="{pad}" width="{W:.1f}" height="{H:.1f}" class="pl"/>'
             f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{pad}" y2="{pad + H:.1f}" class="pl"/>'
             f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{9.15 * sx:.1f}" class="pl"/>'
             f'<rect x="{pad}" y="{cy - pa_w / 2:.1f}" width="{pa_d:.1f}" height="{pa_w:.1f}" class="pl"/>'
             f'<rect x="{pad + W - pa_d:.1f}" y="{cy - pa_w / 2:.1f}" width="{pa_d:.1f}" height="{pa_w:.1f}" class="pl"/>')
    return (f'<svg viewBox="0 0 {width} {H + 2 * pad:.0f}" class="chart pitch" role="img" '
            f'aria-label="Share of ball time per pitch zone">{cells}{lines}</svg>')


EVENT_ROWS = [  # (label, event_type, outcome filter or None)
    ("Passes completed", "pass", "completed"), ("Passes not completed", "pass", "incomplete"),
    ("Crosses", "cross", None), ("Shots", "shot", None), ("Shots on target", "shot", ("goal", "saved")),
    ("Goals (from shots)", "shot", "goal"), ("Corners", "corner", None), ("Free kicks", "free_kick", None),
    ("Throw-ins", "throw_in", None), ("Battles won (tackles)", "tackle", "won"),
    ("Battles lost (tackles)", "tackle", "lost"), ("Interceptions", "interception", None),
    ("Ball recoveries", "ball_recovery", None),
]


def event_counts(reviews: pd.DataFrame) -> list[tuple[str, int]]:
    if reviews is None or reviews.empty:
        return [(lab, 0) for lab, _, _ in EVENT_ROWS]
    ok = reviews[reviews["review_status"].isin(["verified", "corrected"])]
    out = []
    for lab, et, oc in EVENT_ROWS:
        s = ok[ok["event_type"] == et]
        if isinstance(oc, tuple):
            s = s[s["outcome"].isin(oc)]
        elif oc:
            s = s[s["outcome"] == oc]
        out.append((lab, len(s)))
    return out


def _mmss(s):
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


# ----------------------------------------------------------------------------- page
def build_report(repo: Path, run_dir: Path, out: Path | None = None) -> Path:
    repo, run_dir = Path(repo), Path(run_dir)
    state = json.loads((run_dir / "manifest.json").read_text())
    match_id = state["match_id"]
    mpath = repo / state.get("match_manifest", f"config/{match_id}.json")
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    video = repo / state["video"]
    fps = float(manifest.get("normalized_fps") or 25)
    meta = json.loads((run_dir / "detection_meta.json").read_text())
    tracks = pd.read_csv(run_dir / "tracks.csv")
    balls = pd.read_csv(run_dir / "balls.csv")
    cands = pd.read_csv(run_dir / "event_candidates.csv", dtype={"event_id": str}) \
        if (run_dir / "event_candidates.csv").exists() else pd.DataFrame()
    clusters = pd.read_csv(run_dir / "tracklet_team_clusters.csv") \
        if (run_dir / "tracklet_team_clusters.csv").exists() else pd.DataFrame()
    reviews_path = repo / "data" / "annotations" / match_id / "reviewed_events.csv"
    reviews = read_review_csv(reviews_path) if reviews_path.exists() else pd.DataFrame()
    profiles = pd.read_csv(run_dir / "player_profiles.csv", dtype={"player_id": str}) \
        if (run_dir / "player_profiles.csv").exists() else pd.DataFrame()
    cand_params = json.loads((run_dir / "candidate_params.json").read_text()) \
        if (run_dir / "candidate_params.json").exists() else {}

    n_frames = int(meta["frames_processed"])
    duration = n_frames / fps
    ball_thr = float(cand_params.get("ball_threshold", 0.15))
    from .events.ball import clean_balls
    balls, ball_clean = clean_balls(balls[balls["confidence"] >= ball_thr], fps, manifest.get("pitch_roi_polygon"))
    ball_frames = set(balls["frame_index"].astype(int))
    ball_pct = 100.0 * len(ball_frames) / max(n_frames, 1)
    per_frame = tracks.groupby("frame_index").size().reindex(range(n_frames), fill_value=0)
    track_len = tracks.groupby("track_id")["frame_index"].agg(lambda s: (s.max() - s.min() + 1) / fps)
    long_tracks = int((track_len >= 5).sum())

    # review status per candidate
    status = {}
    if len(reviews):
        status = dict(zip(reviews["event_id"], reviews["review_status"]))
    events = []
    for r in cands.itertuples(index=False):
        st = status.get(r.event_id, "unreviewed") or "unreviewed"
        events.append({"t": float(r.video_time_s), "status": st if st in ("verified", "corrected", "rejected") else "unreviewed",
                       "tip": f"{_mmss(r.video_time_s)} · track {r.initiator_track_id} → {r.receiver_track_id} "
                              f"({r.gap_seconds:.1f} s) · {st}"})
    n_conf = sum(e["status"] in ("verified", "corrected") for e in events)
    n_rej = sum(e["status"] == "rejected" for e in events)

    # density images
    frame = _frame(video)
    players = tracks[tracks["class_id"].isin([1, 2])]
    img_all = density_image(frame, players["foot_x"], players["foot_y"], BLUE_RAMP)
    team_imgs = []
    if len(clusters):
        cmap = dict(zip(clusters["track_id"], clusters["cluster"]))
        pc = players.assign(cluster=players["track_id"].map(cmap))
        for k, ramp, label in ((0, BLUE_RAMP, "Kit-colour group 1"), (1, ORANGE_RAMP, "Kit-colour group 2")):
            sub = pc[pc["cluster"] == k]
            n_tr = int(clusters["cluster"].eq(k).sum())
            team_imgs.append((label, n_tr, density_image(frame, sub["foot_x"], sub["foot_y"], ramp, width=1200),
                              "series1" if k == 0 else "series2"))

    best_ball = balls[balls["confidence"] >= ball_thr].sort_values(["frame_index", "confidence"], ascending=[True, False])
    best_ball = best_ball.drop_duplicates("frame_index")
    img_ball = density_image(frame, best_ball["center_x"], best_ball["center_y"], ORANGE_RAMP, sigma_px=18) \
        if len(best_ball) else None
    stacked = any((f.get("normalized") or {}).get("layout") == "stacked_halves"
                  for f in (manifest.get("feeds") or {}).values()) or "stacked" in str(manifest.get("source_note", ""))
    half_html = ""
    if len(best_ball):
        mid = frame.shape[1] / 2
        left = float((best_ball["center_x"] < mid).mean())
        half_html = (f'<div class="split"><div style="flex:{left:.3f}" class="sl"><span>Left half {left * 100:.0f}%</span></div>'
                     f'<div style="flex:{1 - left:.3f}" class="sr"><span>Right half {(1 - left) * 100:.0f}%</span></div></div>'
                     '<p class="lead" style="margin-top:6px">Share of moments the ball was seen on each side of the halfway line, '
                     'as the camera shows it'
                     + (' (the strip around the halfway line appears in both halves of this view)' if stacked else '') + '.</p>')

    # calibrated extras: top-down ball zones and distance while tracked
    pitch_html, dist_html = "", ""
    tf, tr = run_dir / "tracks_field.parquet", run_dir / "transforms.json"
    if tf.exists() and tr.exists():
        from .analytics.movement import ball_zones, player_distances, project_ball_ground, track_distances
        transforms = json.loads(tr.read_text())
        L, Wd = transforms["field_length_m"], transforms["field_width_m"]
        bxy = project_ball_ground(balls, transforms, ball_thr)
        if len(bxy):
            pitch_html = (f"<p class='lead'>Top-down pitch ({L:g} × {Wd:g} m{' — assumed size' if manifest.get('pitch_size_assumed') else ''}). "
                          f"Based on {len(bxy):,} ball sightings; zones, not exact spots, because a ball in the air "
                          "looks further away than it is.</p>" + pitch_zones_svg(ball_zones(bxy, L, Wd), L, Wd))
        per_track = track_distances(pd.read_parquet(tf), fps)
        assign_p = run_dir / "tracklet_assignments.csv"
        per_player = player_distances(per_track, pd.read_csv(assign_p, dtype={"player_id": str})
                                      if assign_p.exists() else None)
        if len(per_player):
            body = "".join(f"<tr><td>{_esc(r.player_id)}</td><td class='num'>{r.distance_m:,.0f} m</td>"
                           f"<td class='num'>{_mmss(r.tracked_seconds)}</td><td class='num'>{r.tracklets}</td></tr>"
                           for r in per_player.itertuples(index=False))
            dist_html = ("<table><thead><tr><th>Player</th><th>Distance while tracked</th><th>Time tracked</th>"
                         f"<th>Track pieces</th></tr></thead><tbody>{body}</tbody></table>")
        elif len(per_track):
            top = per_track.sort_values("distance_m", ascending=False).head(12)
            body = "".join(f"<tr><td>#{r.track_id}</td><td class='num'>{r.distance_m:,.0f} m</td>"
                           f"<td class='num'>{_mmss(r.tracked_seconds)}</td><td class='num'>"
                           f"{'—' if r.top_speed_ms is None else f'{r.top_speed_ms * 3.6:.0f} km/h'}</td></tr>"
                           for r in top.itertuples(index=False))
            dist_html = ("<p class='lead'>Per tracker ID, because no identities have been confirmed yet: one player can "
                         "appear as several IDs, so these are pieces of runs, not match totals.</p><table><thead><tr>"
                         "<th>Tracker ID</th><th>Distance while tracked</th><th>Time tracked</th><th>Fast running (95th pct)</th>"
                         f"</tr></thead><tbody>{body}</tbody></table>")
    if not pitch_html:
        pitch_html = ("<p class='empty'>Needs pitch calibration: mark at least 14 pitch points on a frame once for this "
                      "camera position (docs/VEO.md). An assumed 11v11 size (105 × 68 m) can be used for the pitch "
                      "dimensions, but the points must come from the video.</p>")
    if not dist_html:
        dist_html = ("<p class='empty'>Needs pitch calibration (to turn pixels into metres) and confirmed player "
                     "identities (each player currently splits into many short tracker IDs).</p>")
    ev_counts = event_counts(reviews)
    ev_html = "".join(f"<tr><td>{_esc(lab)}</td><td class='num'>{n}</td></tr>" for lab, n in ev_counts)

    t_axis = np.arange(n_frames) / fps
    step = max(1, n_frames // 400)
    line = line_chart(t_axis[::step], per_frame.values[::step], "people-title", "people detected", events=events)
    strip = strip_chart(t_axis, [i in ball_frames for i in range(n_frames)], events, duration)
    hist_bins = [0, 1, 2, 5, 10, 20, 30, 45, 60] if duration > 30 else [0, 0.5, 1, 2, 5, 10, 20, 30]
    hist_bins = [b for b in hist_bins if b <= max(duration, 1)] + ([duration + 0.01] if hist_bins[-1] < duration else [])
    hist = histogram(track_len.values, sorted(set(hist_bins)), xlabel="seconds a player kept the same ID")

    calibrated = (run_dir / "transforms.json").exists()
    det_name = Path(str(state.get("config", {}).get("detection", {}).get("player_weights", ""))).name
    smoke = "smoke" in det_name or "soccernet-yolov8" in det_name

    def chip(ok, text):
        cls = "ok" if ok else "warn"
        icon = "✓" if ok else "!"
        return f'<span class="chip {cls}"><span aria-hidden="true">{icon}</span> {_esc(text)}</span>'

    chips = "".join([
        chip(manifest.get("video_use_permission") == "granted", "Video permission recorded"),
        chip(calibrated, "Pitch calibrated" if calibrated else "No pitch calibration: positions are camera view, not metres"),
        chip(not smoke, "Soccer detection models" if not smoke else "Stand-in detector (trial)"),
        chip(n_conf > 0, f"{n_conf} of {len(events)} possible passes confirmed" if events else "No possible passes found"),
    ])

    rows = ""
    for r in cands.itertuples(index=False):
        st = status.get(r.event_id, "unreviewed") or "unreviewed"
        clip = run_dir / "highlights" / f"{r.event_id}.mp4"
        rel = clip.relative_to(run_dir).as_posix() if clip.exists() else None
        rows += (f"<tr><td>{_mmss(r.video_time_s)}</td><td>#{r.initiator_track_id} → #{r.receiver_track_id}</td>"
                 f"<td class='num'>{r.gap_seconds:.1f} s</td><td><span class='st st-{_esc(st)}'>{_esc(st)}</span></td>"
                 f"<td>{f'<a href={chr(34)}{rel}{chr(34)}>clip</a>' if rel else '—'}</td></tr>")
    if not rows:
        rows = "<tr><td colspan='5' class='empty'>No possible passes in this run. Zero here is not proof of zero passes — check ball visibility above.</td></tr>"

    if len(profiles):
        cols = ["player_id", "total_reviewed_events", "completed_passes", "pass_attempts_resolved",
                "passing_accuracy_pct", "shots", "goals", "ball_recoveries", "interceptions", "tackles_won"]
        heads = ["Player", "Reviewed events", "Passes completed", "Passes (known outcome)", "Pass %",
                 "Shots", "Goals", "Recoveries", "Interceptions", "Tackles won"]
        prof_html = "<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in heads) + "</tr></thead><tbody>"
        for r in profiles[cols].itertuples(index=False):
            prof_html += "<tr>" + "".join(
                f"<td class='{'num' if i else ''}'>{'—' if pd.isna(v) else _esc(v)}</td>" for i, v in enumerate(r)) + "</tr>"
        prof_html += "</tbody></table>"
    else:
        prof_html = ("<p class='empty'>No player statistics yet. They appear only after a reviewer confirms events "
                     "and names the player, so nothing unreviewed is ever counted.</p>")

    team_html = ""
    for label, n_tr, img, cls in team_imgs:
        team_html += (f'<figure class="half"><figcaption><span class="key {cls}"></span>{_esc(label)} '
                      f'<span class="muted">· {n_tr} tracks</span></figcaption><img src="{img}" alt="Where {label} players were"></figure>')

    quality = [
        f"Analysed {duration:.0f} s at {fps:g} frames per second ({n_frames} frames).",
        f"The tracker issued {meta['distinct_track_ids']} player IDs; {long_tracks} lasted 5 s or more. "
        "Players lose their ID after occlusions, so per-player tracking across the clip is not reliable yet.",
        f"The ball was seen clearly in {ball_pct:.0f}% of frames; passes are only proposed where it is seen. "
        f"{ball_clean['static_removed']} detections of a motionless object and {ball_clean['outside_pitch_removed']} "
        "off-pitch detections were not counted as the ball.",
    ]
    if meta.get("person_detections_outside_pitch_roi_dropped"):
        quality.append(f"{meta['person_detections_outside_pitch_roi_dropped']:,} detections outside the pitch outline "
                       "(spectators, benches) were removed before tracking.")
    if not calibrated:
        quality.append("Without pitch calibration the maps show the camera's view, not a top-down pitch; "
                       "distances and speeds are not computed.")
    if smoke:
        quality.append("A stand-in detector trained on TV broadcasts was used; the soccer models from the guide should do better.")
    quality_html = "".join(f"<li>{_esc(q)}</li>" for q in quality)

    table_rows = "".join(f"<tr><td>{_mmss(t)}</td><td class='num'>{int(v)}</td><td>{'yes' if int(round(t * fps)) in ball_frames else 'no'}</td></tr>"
                         for t, v in zip(t_axis[::max(1, int(fps))], per_frame.values[::max(1, int(fps))]))

    title = f"Match analysis · {match_id}"
    src_note = manifest.get("source_note") or manifest.get("camera_type") or ""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page = TEMPLATE.format(
        title=_esc(title), match_id=_esc(match_id), src_note=_esc(src_note), generated=generated,
        run_id=_esc(state["run_id"]), chips=chips,
        k_dur=_mmss(duration), k_people=f"{per_frame.median():.0f}", k_ball=f"{ball_pct:.0f}%",
        k_pass=str(len(events)), k_pass_sub=f"{n_conf} confirmed · {n_rej} rejected · {len(events) - n_conf - n_rej} to review",
        img_all=img_all, team_html=team_html, line=line, strip=strip, hist=hist, rows=rows, prof_html=prof_html,
        quality_html=quality_html, table_rows=table_rows, n_tracks=meta["distinct_track_ids"], long_tracks=long_tracks,
        ball_img=(f'<figure><img src="{img_ball}" alt="Where the ball was seen"></figure>' if img_ball else
                  "<p class='empty'>The ball was not detected clearly enough to map.</p>"),
        half_html=half_html, pitch_html=pitch_html, dist_html=dist_html, ev_html=ev_html,
        n_confirmed=int(reviews["review_status"].isin(["verified", "corrected"]).sum()) if len(reviews) else 0)
    out = Path(out or run_dir / "report.html")
    out.write_text(page, encoding="utf-8")
    return out


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  color-scheme: light;
  --bg:#f6f6f4; --surface-1:#fcfcfb; --surface-2:#f0efec; --border:#e2e1dc;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#76756f;
  --series-1:#2a78d6; --series-2:#eb6834; --grid:#e8e7e2; --accent:#1c5cab;
  --good:#1a7f37; --warn:#9a6700; --bad:#c22f2f; --seen:#2a78d6; --unseen:#e8e7e2;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --bg:#121211; --surface-1:#1a1a19; --surface-2:#232321; --border:#2f2f2c;
    --text-primary:#f0efec; --text-secondary:#c3c2b7; --text-muted:#99988f;
    --series-1:#3987e5; --series-2:#d95926; --grid:#2b2b29; --accent:#86b6ef;
    --good:#4ac26b; --warn:#d4a72c; --bad:#ff7b72; --seen:#3987e5; --unseen:#2b2b29;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --bg:#121211; --surface-1:#1a1a19; --surface-2:#232321; --border:#2f2f2c;
  --text-primary:#f0efec; --text-secondary:#c3c2b7; --text-muted:#99988f;
  --series-1:#3987e5; --series-2:#d95926; --grid:#2b2b29; --accent:#86b6ef;
  --good:#4ac26b; --warn:#d4a72c; --bad:#ff7b72; --seen:#3987e5; --unseen:#2b2b29;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text-primary);
  font:15px/1.5 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:32px 16px 64px; }}
header h1 {{ font-size:28px; line-height:1.2; margin:0 0 4px; letter-spacing:-0.01em; }}
.sub {{ color:var(--text-secondary); margin:0 0 16px; }}
.muted {{ color:var(--text-muted); }}
.chips {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0 24px; }}
.chip {{ display:inline-flex; gap:6px; align-items:center; padding:4px 10px; border-radius:999px; font-size:13px;
  background:var(--surface-1); border:1px solid var(--border); color:var(--text-secondary); }}
.chip.ok span {{ color:var(--good); font-weight:700; }} .chip.warn span {{ color:var(--warn); font-weight:700; }}
.kpis {{ display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:12px; margin-bottom:24px; }}
.kpi {{ background:var(--surface-1); border:1px solid var(--border); border-radius:12px; padding:16px; }}
.kpi .label {{ font-size:13px; color:var(--text-secondary); }}
.kpi .value {{ font-size:32px; font-weight:650; letter-spacing:-0.02em; font-variant-numeric:tabular-nums; }}
.kpi .note {{ font-size:12px; color:var(--text-muted); }}
section {{ background:var(--surface-1); border:1px solid var(--border); border-radius:14px; padding:20px; margin-bottom:20px; }}
section h2 {{ font-size:18px; margin:0 0 4px; }}
section p.lead {{ margin:0 0 14px; color:var(--text-secondary); font-size:14px; }}
figure {{ margin:0; }} img {{ width:100%; height:auto; display:block; border-radius:10px; }}
.teams {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px; }}
figcaption {{ font-size:13px; color:var(--text-secondary); margin:0 0 6px; display:flex; align-items:center; gap:6px; }}
.key {{ width:12px; height:12px; border-radius:3px; display:inline-block; }}
.key.series1 {{ background:var(--series-1); }} .key.series2 {{ background:var(--series-2); }}
.chart {{ width:100%; height:auto; display:block; overflow:visible; }}
.chart .grid {{ stroke:var(--grid); stroke-width:1; }}
.chart .tick {{ fill:var(--text-muted); font-size:11px; font-variant-numeric:tabular-nums; }}
.chart .series1 {{ stroke:var(--series-1); stroke-width:2; stroke-linejoin:round; }}
.chart .series1fill {{ fill:var(--series-1); }}
.chart .evline {{ stroke:var(--text-muted); stroke-width:1; stroke-dasharray:3 3; }}
.chart .cross {{ stroke:var(--text-muted); stroke-width:1; }}
.chart .crossdot {{ fill:var(--series-1); stroke:var(--surface-1); stroke-width:2; }}
.strip .seen {{ fill:var(--seen); }} .strip .unseen {{ fill:var(--unseen); }}
.ev-unreviewed {{ fill:var(--surface-1); stroke:var(--text-secondary); stroke-width:2; }}
.ev-verified, .ev-corrected {{ fill:var(--good); stroke:var(--surface-1); stroke-width:2; }}
.ev-rejected {{ fill:var(--text-muted); stroke:var(--surface-1); stroke-width:2; opacity:.6; }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; font-size:12px; color:var(--text-secondary); margin:6px 0 0 44px; }}
.legend i {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:-1px; }}
.two {{ display:grid; grid-template-columns:1.3fr 1fr; gap:20px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--border); }}
th {{ font-size:12px; color:var(--text-secondary); font-weight:600; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.empty {{ color:var(--text-muted); font-style:italic; }}
.st {{ font-size:12px; padding:2px 8px; border-radius:999px; border:1px solid var(--border); }}
.st-verified, .st-corrected {{ color:var(--good); }} .st-rejected {{ color:var(--text-muted); }} .st-unreviewed {{ color:var(--warn); }}
a {{ color:var(--accent); }}
ul.quality {{ margin:0; padding-left:20px; color:var(--text-secondary); font-size:14px; }}
ul.quality li {{ margin:4px 0; }}
details {{ margin-top:10px; font-size:13px; color:var(--text-secondary); }}
details table {{ max-width:420px; }}
#tip {{ position:fixed; pointer-events:none; background:var(--text-primary); color:var(--surface-1); font-size:12px;
  padding:5px 8px; border-radius:6px; opacity:0; transition:opacity .08s; z-index:10; white-space:nowrap; }}
footer {{ color:var(--text-muted); font-size:12px; margin-top:24px; }}
.split {{ display:flex; gap:2px; height:28px; margin-top:12px; font-size:12px; }}
.split > div {{ display:flex; align-items:center; padding:0 10px; border-radius:4px; min-width:90px; }}
.split .sl {{ background:var(--series-2); color:#fff; }} .split .sr {{ background:var(--surface-2); color:var(--text-primary); justify-content:flex-end; }}
.chart.pitch .pl {{ fill:none; stroke:var(--text-muted); stroke-width:1.5; }}
.chart.pitch .zlab {{ font-size:12px; font-weight:600; font-variant-numeric:tabular-nums; }}
@media (max-width:760px) {{ .kpis {{ grid-template-columns:1fr 1fr; }} .teams, .two {{ grid-template-columns:1fr; }} }}
</style></head>
<body><main>
<header>
  <h1>{title}</h1>
  <p class="sub">{src_note}</p>
  <div class="chips">{chips}</div>
</header>

<div class="kpis">
  <div class="kpi"><div class="label">Video analysed</div><div class="value">{k_dur}</div><div class="note">minutes : seconds</div></div>
  <div class="kpi"><div class="label">People on the pitch</div><div class="value">{k_people}</div><div class="note">median per frame</div></div>
  <div class="kpi"><div class="label">Ball clearly seen</div><div class="value">{k_ball}</div><div class="note">of frames</div></div>
  <div class="kpi"><div class="label">Possible passes</div><div class="value">{k_pass}</div><div class="note">{k_pass_sub}</div></div>
</div>

<section>
  <h2>Where the players were</h2>
  <p class="lead">Darker blue = more time spent there, across all players, over the full camera view.</p>
  <figure><img src="{img_all}" alt="Density of player positions over the panorama"></figure>
  <div class="teams">{team_html}</div>
  <p class="lead" style="margin-top:10px">Groups come from shirt colour and are not yet matched to team names. Referees and goalkeepers are excluded.</p>
</section>

<section>
  <h2>Where the ball was</h2>
  <p class="lead">Darker orange = the ball was seen there more often (camera view).</p>
  {ball_img}
  {half_html}
</section>

<div class="two">
<section>
  <h2>Ball time by pitch zone</h2>
  {pitch_html}
</section>
<section>
  <h2>Distance run</h2>
  {dist_html}
</section>
</div>

<section>
  <h2>Play over time</h2>
  <p class="lead">People detected on the pitch each moment, ball visibility, and where possible passes were found. Hover for details.</p>
  {line}
  {strip}
  <div class="legend"><span><i style="background:var(--seen);border-radius:2px"></i>ball seen</span>
    <span><i style="background:var(--surface-1);border:2px solid var(--text-secondary)"></i>possible pass, to review</span>
    <span><i style="background:var(--good)"></i>confirmed</span><span><i style="background:var(--text-muted)"></i>rejected</span></div>
  <details><summary>Show as a table</summary><table><thead><tr><th>Time</th><th>People</th><th>Ball seen</th></tr></thead>
  <tbody>{table_rows}</tbody></table></details>
</section>

<div class="two">
<section>
  <h2>Possible passes</h2>
  <p class="lead">Found by the computer and waiting for a person to confirm. They never count as statistics until confirmed.</p>
  <table><thead><tr><th>Time</th><th>From → to (tracker IDs)</th><th>Gap</th><th>Status</th><th>Video</th></tr></thead>
  <tbody>{rows}</tbody></table>
</section>
<section>
  <h2>How long players kept their ID</h2>
  <p class="lead">{n_tracks} IDs issued; {long_tracks} lasted 5 seconds or more. Short bars on the left mean players were often re-identified as someone new.</p>
  {hist}
</section>
</div>

<div class="two">
<section>
  <h2>Match events</h2>
  <p class="lead">Confirmed by a reviewer ({n_confirmed} so far). Events are labelled in the review app; the computer only proposes ball transfers.</p>
  <table><tbody>{ev_html}</tbody></table>
</section>
<section>
  <h2>Player statistics</h2>
  {prof_html}
</section>
</div>

<section>
  <h2>How much to trust this</h2>
  <ul class="quality">{quality_html}</ul>
</section>

<footer>Run {run_id} · generated {generated} · Soccer Vision</footer>
</main>
<div id="tip" role="tooltip"></div>
<script>
(function() {{
  const tip = document.getElementById('tip');
  const show = (e, text) => {{ tip.textContent = text; tip.style.opacity = 1;
    tip.style.left = Math.min(e.clientX + 12, window.innerWidth - tip.offsetWidth - 8) + 'px';
    tip.style.top = (e.clientY - 34) + 'px'; }};
  const hide = () => {{ tip.style.opacity = 0; }};
  document.querySelectorAll('[data-tip]').forEach(el => {{
    el.addEventListener('mousemove', e => show(e, el.dataset.tip)); el.addEventListener('mouseleave', hide); }});
  document.querySelectorAll('svg[data-series]').forEach(svg => {{
    const data = JSON.parse(svg.dataset.series), x0 = +svg.dataset.x0, x1 = +svg.dataset.x1, tmax = +svg.dataset.tmax;
    const cross = svg.querySelector('.cross'), dot = svg.querySelector('.crossdot'), hit = svg.querySelector('.hit');
    const pts = svg.querySelector('polyline').getAttribute('points').split(' ').map(p => p.split(',').map(Number));
    hit.addEventListener('mousemove', e => {{
      const r = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal;
      const x = (e.clientX - r.left) * vb.width / r.width;
      const t = Math.max(0, Math.min(tmax, (x - x0) / (x1 - x0) * tmax));
      let i = 0, best = 1e9; data.forEach((d, k) => {{ const dd = Math.abs(d[0] - t); if (dd < best) {{ best = dd; i = k; }} }});
      const [px, py] = pts[i];
      cross.setAttribute('x1', px); cross.setAttribute('x2', px); cross.setAttribute('visibility', 'visible');
      dot.setAttribute('cx', px); dot.setAttribute('cy', py); dot.setAttribute('visibility', 'visible');
      const s = Math.round(data[i][0]);
      show(e, Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0') + ' · ' + data[i][1] + ' ' + svg.dataset.ylabel);
    }});
    hit.addEventListener('mouseleave', () => {{ hide(); cross.setAttribute('visibility', 'hidden'); dot.setAttribute('visibility', 'hidden'); }});
  }});
}})();
</script>
</body></html>
"""
