"""Stage implementations for the orchestrated pipeline. Each returns {"outputs", "records", "notes"}."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..manifest import validate_manifest
from ..provenance import video_summary
from ..schemas import read_balls, read_tracks
from .runner import Context, StageSkipped, _abs


def _prev(ctx: Context, stage: str, name: str) -> Path | None:
    return ctx.outputs.get(stage, {}).get(name)


def stage_inputs(name: str, ctx: Context) -> dict:
    """Files whose checksums gate reuse of a stage."""
    det = ctx.config["detection"]
    m = {
        "validate": {"video": ctx.video},
        "detect_track": {"video": ctx.video, "class_map": _abs(ctx.repo, det["class_map"])},
        "calibrate": {},
        "identify": {"tracks": _prev(ctx, "detect_track", "tracks")},
        "candidates": {"tracks": _prev(ctx, "detect_track", "tracks"), "balls": _prev(ctx, "detect_track", "balls")},
        "temporal_model": {},
        "highlights": {"events": _prev(ctx, "candidates", "event_candidates")},
        "review_queue": {"events": _prev(ctx, "candidates", "event_candidates")},
        "aggregate": {"reviewed": ctx.annotations_dir / "reviewed_events.csv"},
        "report": {"reviewed": ctx.annotations_dir / "reviewed_events.csv",
                   "events": _prev(ctx, "candidates", "event_candidates")},
    }[name]
    lm = ctx.manifest.get("calibration_landmarks")
    if name == "calibrate" and lm:
        m["landmarks"] = _abs(ctx.repo, lm)
    if name == "highlights" and ctx.manifest_path is not None:
        m["match_manifest"] = ctx.manifest_path  # follow-cam offset lives here: changing it re-cuts clips
    return {k: v for k, v in m.items() if v is not None}


def stage_validate(ctx: Context):
    rep = validate_manifest(ctx.manifest, production=ctx.config.get("require_production_manifest", False))
    summary = video_summary(ctx.video)
    errors = list(rep.errors)
    fps = ctx.manifest["normalized_fps"]
    if not summary["constant_frame_rate"]:
        errors.append(f"video is not constant frame rate: {summary}")
    if summary["avg_frame_rate"] and abs(summary["avg_frame_rate"] - fps) > 1e-3:
        errors.append(f"video FPS {summary['avg_frame_rate']} != manifest normalized_fps {fps}")
    if errors:
        raise ValueError("; ".join(errors))
    out = ctx.path("video_info.json")
    out.write_text(json.dumps({"summary": summary, "manifest_warnings": rep.warnings}, indent=2))
    return {"outputs": {"video_info": out}, "records": 1,
            "notes": "; ".join(rep.warnings) if rep.warnings else "manifest ok"}


def stage_detect_track(ctx: Context):
    from ..detection.classmap import ClassMapConfig
    from ..detection.export import ExportConfig, UltralyticsPredictor, export_detections, weights_info
    det = ctx.config["detection"]
    cm = ClassMapConfig.load(_abs(ctx.repo, det["class_map"]))
    pw, bw = _abs(ctx.repo, det["player_weights"]), _abs(ctx.repo, det.get("ball_weights"))
    device = ctx.config["device"]
    person = UltralyticsPredictor(pw, device, det["imgsz"], min(det["person_conf"], det["ball_conf_keep"]))
    ball = UltralyticsPredictor(bw, device, det["imgsz"], det["ball_conf_keep"]) if bw else None
    tiling = det.get("tiling") or {}
    if tiling.get("enabled"):
        from ..detection.tiling import TiledPredictor
        person = TiledPredictor(person, tiling.get("tile_size", 1280), tiling.get("overlap", 0.2))
        ball = TiledPredictor(ball, tiling.get("tile_size", 1280), tiling.get("overlap", 0.2)) if ball else None
    tr = det.get("tracker", {})
    cfg = ExportConfig(match_id=ctx.match_id, video_id=ctx.manifest["video_id"], run_id=ctx.run_id,
                       fps=float(ctx.manifest["normalized_fps"]), person_conf=det["person_conf"],
                       ball_conf_keep=det["ball_conf_keep"], max_frames=det.get("max_frames"),
                       roi_polygon=ctx.manifest.get("pitch_roi_polygon"), **tr)
    annotated = ctx.path("tracking_annotated.mp4") if det.get("annotated_video") else None
    meta = export_detections(ctx.video, ctx.run_dir, cfg, person, cm, ball, annotated_video=annotated,
                             weights_info=weights_info(pw, bw))
    outs = {"tracks": ctx.path("tracks.csv"), "balls": ctx.path("balls.csv"), "meta": ctx.path("detection_meta.json")}
    if annotated:
        outs["annotated_video"] = annotated
    return {"outputs": outs, "records": meta["track_rows"],
            "notes": f"{meta['frames_processed']} frames, {meta['distinct_track_ids']} track IDs, "
                     f"ball candidate in {meta['frames_with_ball_candidate']} frames"}


def stage_calibrate(ctx: Context):
    from ..calibration.homography import calibrate, project_tracks
    lm = ctx.manifest.get("calibration_landmarks")
    if not lm:
        raise StageSkipped("no calibration_landmarks in match manifest: field coordinates are explicitly MISSING")
    c = ctx.config.get("calibration", {})
    transforms = calibrate(_abs(ctx.repo, lm), c.get("max_median_error_m", 2.0), c.get("accept_unvalidated", False))
    for key in ("field_length_m", "field_width_m"):
        mv = ctx.manifest.get(key)
        if mv is not None and abs(float(mv) - float(transforms[key])) > 0.01:
            raise ValueError(f"{key}: landmark file {transforms[key]} != manifest {mv}")
    tp = ctx.path("transforms.json")
    tp.write_text(json.dumps(transforms, indent=2))
    tracks = read_tracks(_prev(ctx, "detect_track", "tracks"))
    proj = project_tracks(tracks, transforms)
    pq = ctx.path("tracks_field.parquet")
    proj.to_parquet(pq, index=False)
    ok = sum(s["status"] == "calibrated" for s in transforms["segments"])
    return {"outputs": {"transforms": tp, "tracks_field": pq}, "records": int(proj["calibrated"].sum()),
            "notes": f"{ok}/{len(transforms['segments'])} segments accepted; "
                     f"{(~proj['calibrated']).sum()} track rows uncalibrated"}


def stage_identify(ctx: Context):
    from ..identity.assignments import ASSIGNMENT_COLUMNS, sharpest_crops
    from ..identity.team import cluster_teams
    idc = ctx.config.get("identity", {})
    tracks = read_tracks(_prev(ctx, "detect_track", "tracks"))
    outs = {}
    if tracks.empty:
        raise StageSkipped("no tracks to identify")
    if idc.get("team_clustering", True):
        teams = cluster_teams(ctx.video, tracks, idc.get("samples_per_track", 8))
        tp = ctx.path("tracklet_team_clusters.csv")
        teams.to_csv(tp, index=False)
        outs["team_clusters"] = tp
    crops = sharpest_crops(ctx.video, tracks, ctx.path("identity_crops"), idc.get("review_crops_per_track", 5))
    cp = ctx.path("identity_review_queue.csv")
    crops.to_csv(cp, index=False)
    outs["review_queue"] = cp
    # Empty, schema-correct assignment template: identities are assigned by humans (or future OCR/re-ID
    # evidence) and then validated with `soccer-vision identity validate`.
    tmpl = pd.DataFrame({"video_id": ctx.manifest["video_id"], "run_id": ctx.run_id, "segment_id": 0,
                         "track_id": sorted(tracks["track_id"].dropna().astype(int).unique())})
    for c in ASSIGNMENT_COLUMNS:
        if c not in tmpl:
            tmpl[c] = "unreviewed" if c == "review_status" else ""
    ap = ctx.path("tracklet_assignments_template.csv")
    tmpl[ASSIGNMENT_COLUMNS].to_csv(ap, index=False)
    outs["assignments_template"] = ap
    return {"outputs": outs, "records": int(tracks["track_id"].nunique()),
            "notes": "team clusters are unmapped cluster indices; roster identities require human review"}


def stage_candidates(ctx: Context):
    from ..events.candidates import CandidateParams, generate_with_audit, params_dict
    p = CandidateParams(fps=float(ctx.manifest["normalized_fps"]), **ctx.config.get("candidates", {}))
    tracks = read_tracks(_prev(ctx, "detect_track", "tracks"))
    balls = read_balls(_prev(ctx, "detect_track", "balls"))
    from ..events.ball import clean_balls
    balls, ball_stats = clean_balls(balls[balls["confidence"] >= p.ball_threshold], p.fps,
                                    ctx.manifest.get("pitch_roi_polygon"))
    contacts, events, suppressed = generate_with_audit(tracks, balls, ctx.match_id, p,
                                                       video_id=ctx.manifest["video_id"], run_id=ctx.run_id)
    outs = {"contact_candidates": ctx.path("contact_candidates.csv"),
            "event_candidates": ctx.path("event_candidates.csv"),
            "suppressed_candidates": ctx.path("suppressed_candidates.csv"),
            "candidate_params": ctx.path("candidate_params.json")}
    contacts.to_csv(outs["contact_candidates"], index=False)
    events.to_csv(outs["event_candidates"], index=False)
    suppressed.to_csv(outs["suppressed_candidates"], index=False)
    outs["candidate_params"].write_text(json.dumps({**params_dict(p), "ball_cleaning": ball_stats}, indent=2))
    return {"outputs": outs, "records": len(events),
            "notes": f"{len(events)} unverified transfer candidates; {len(suppressed)} suppressed with reasons; "
                     f"{ball_stats['static_removed']} static and {ball_stats['outside_pitch_removed']} off-pitch ball "
                     "detections removed; "
                     f"{len(contacts)} ball frames evaluated"}


def stage_temporal_model(ctx: Context):
    if not ctx.config.get("temporal_model", {}).get("enabled"):
        raise StageSkipped("temporal event model not enabled: no trained model with held-out whole-match "
                           "evaluation exists (Steps 11/26)")
    raise NotImplementedError("temporal_model.enabled=true but no model integration is implemented")


def stage_highlights(ctx: Context):
    from ..events.highlights import make_highlights
    ev = pd.read_csv(_prev(ctx, "candidates", "event_candidates"), dtype={"event_id": str})
    h = ctx.config.get("highlights", {})
    out_dir = ctx.path("highlights")
    index = make_highlights(ctx.video, ev, out_dir, h.get("before", 4.0), h.get("after", 6.0))
    outs = {"clip_index": out_dir / "clip_index.csv"}
    notes = f"{len(index)} clips from the analysis feed"
    # Veo: also cut clips from the follow-cam ("standard") feed, which is easier for reviewers to watch.
    m = ctx.manifest
    feeds = m.get("feeds") or {}
    review = m.get("review_feed")
    if review and review != m.get("analysis_feed") and review in feeds:
        offset = m.get("standard_offset_seconds")
        if offset is None or not m.get("standard_offset_verified"):
            notes += ("; follow-cam clips NOT cut: set and verify the follow-cam time offset "
                      "(soccer-vision feed-offset) first")
        else:
            rv = ctx.repo / feeds[review]["normalized"]["path"]
            rdir = ctx.path("highlights_review")
            ridx = make_highlights(rv, ev, rdir, h.get("before", 4.0), h.get("after", 6.0),
                                   time_offset=float(offset), skip_outside=True)
            outs["review_clip_index"] = rdir / "clip_index.csv"
            notes += f"; {len(ridx)} follow-cam clips (offset {offset:+.2f} s)"
    return {"outputs": outs, "records": len(index), "notes": notes}


def stage_review_queue(ctx: Context):
    from ..review.store import ReviewStore
    ev = pd.read_csv(_prev(ctx, "candidates", "event_candidates"), dtype={"event_id": str, "match_id": str})
    store = ReviewStore(ctx.annotations_dir, ctx.match_id)
    res = store.sync_candidates(ev)
    df = store.load()
    pending = int((df["review_status"] == "unreviewed").sum())
    return {"outputs": {}, "records": res["total"],
            "notes": f"review store {ctx.rel(store.path)}: {res['added']} new, {pending} awaiting human review, "
                     f"{res['orphaned']} orphaned"}


def stage_aggregate(ctx: Context):
    from ..analytics.profiles import build_profiles
    from ..schemas import read_review_csv
    rp = ctx.annotations_dir / "reviewed_events.csv"
    if not rp.exists():
        raise StageSkipped("no review store yet")
    ev = read_review_csv(rp)
    profiles, report = build_profiles(ev)
    if report["confirmed"] == 0:
        raise StageSkipped("no verified/corrected events yet: no player statistics are produced")
    pp, rpj = ctx.path("player_profiles.csv"), ctx.path("profile_coverage_report.json")
    profiles.to_csv(pp, index=False)
    rpj.write_text(json.dumps(report, indent=2))
    return {"outputs": {"player_profiles": pp, "coverage_report": rpj}, "records": len(profiles),
            "notes": f"{report['attributed_to_players']} attributed confirmed events; "
                     f"{report['excluded_not_confirmed']} excluded as not confirmed"}


def stage_report(ctx: Context):
    from ..report import build_report
    out = build_report(ctx.repo, ctx.run_dir)
    return {"outputs": {"report": out}, "records": 1, "notes": f"open {ctx.rel(out)} in a web browser"}
