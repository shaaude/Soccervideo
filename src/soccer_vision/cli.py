"""`soccer-vision` command line. Every stage is also runnable on its own (guide CLIs preserved in scripts/)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def _repo() -> Path:
    env = os.environ.get("SOCCER_VISION_ROOT")
    if env:
        return Path(env).resolve()
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "config" / "pipeline.yaml").exists():
            return p
    return here


def cmd_normalize(a):
    from .normalize import normalize_video
    rec = normalize_video(Path(a.input), Path(a.output), a.fps, a.max_width, a.crf, a.clip_seconds, a.start_seconds)
    print(json.dumps({"normalized": rec["normalized"]["path"], "sha256": rec["normalized"]["sha256"],
                      "fps": rec["normalized"]["avg_frame_rate"], "duration_s": rec["normalized"]["duration_seconds"]}, indent=2))


def cmd_validate_manifest(a):
    from .manifest import load_manifest, validate_manifest
    rep = validate_manifest(load_manifest(a.manifest), production=a.production, repo_root=_repo() if a.check_files else None)
    for w in rep.warnings:
        print("WARNING:", w)
    for e in rep.errors:
        print("ERROR:", e)
    print("OK" if rep.ok else "INVALID")
    return 0 if rep.ok else 1


def cmd_export(a):
    from .detection.classmap import ClassMapConfig
    from .detection.export import ExportConfig, UltralyticsPredictor, export_detections, weights_info
    device = a.device or os.environ.get("SOCCER_DEVICE", "cpu")
    fps = a.fps
    if fps is None:
        from .provenance import video_summary
        s = video_summary(a.video)
        if not s["constant_frame_rate"]:
            print("ERROR: video is not constant frame rate; normalize it first", file=sys.stderr)
            return 2
        fps = s["avg_frame_rate"]
    cm = ClassMapConfig.load(a.class_map)
    person = UltralyticsPredictor(Path(a.player_weights), device, a.imgsz, min(a.person_conf, a.ball_conf_keep))
    ball = UltralyticsPredictor(Path(a.ball_weights), device, a.imgsz, a.ball_conf_keep) if a.ball_weights else None
    video_id = a.video_id or f"{a.match_id}-{Path(a.video).stem}"
    cfg = ExportConfig(a.match_id, video_id, a.run_id or "adhoc", float(fps), a.person_conf, a.ball_conf_keep, a.max_frames)
    meta = export_detections(Path(a.video), Path(a.out_dir), cfg, person, cm, ball,
                             annotated_video=Path(a.annotated_video) if a.annotated_video else None,
                             weights_info=weights_info(a.player_weights, a.ball_weights))
    print(json.dumps({k: meta[k] for k in ("frames_processed", "track_rows", "distinct_track_ids", "ball_rows",
                                           "frames_with_ball_candidate", "elapsed_seconds")}, indent=2))
    print("Track IDs are temporary tracker IDs, not player identities.")


def cmd_candidates(a):
    from .events.candidates import CandidateParams, generate_with_audit, load_inputs, params_dict
    tracks, balls = load_inputs(a.tracks, a.balls, a.match_id)
    p = CandidateParams(fps=a.fps)
    contacts, events, suppressed = generate_with_audit(tracks, balls, a.match_id, p)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    contacts.to_csv(out / "contact_candidates.csv", index=False)
    events.to_csv(out / "event_candidates.csv", index=False)
    suppressed.to_csv(out / "suppressed_candidates.csv", index=False)
    (out / "candidate_params.json").write_text(json.dumps(params_dict(p), indent=2))
    print(f"Observed ball frames: {len(contacts)}; transfer candidates: {len(events)}; suppressed: {len(suppressed)}")
    print("These are unverified hypotheses, not completed passes.")


def cmd_highlights(a):
    from .events.highlights import make_highlights
    idx = make_highlights(Path(a.video), pd.read_csv(a.events, dtype={"event_id": str}), Path(a.out_dir), a.before, a.after)
    print(f"Created {len(idx)} review clips; index: {Path(a.out_dir) / 'clip_index.csv'}")


def cmd_profiles(a):
    from .analytics.profiles import ProfileError, build_profiles
    from .schemas import read_review_csv
    exposure = pd.read_csv(a.exposure, dtype={"player_id": str, "match_id": str}) if a.exposure else None
    try:
        profiles, report = build_profiles(read_review_csv(a.events), exposure=exposure)
    except ProfileError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(out, index=False)
    out.with_suffix(".coverage.json").write_text(json.dumps(report, indent=2))
    print(f"Created {len(profiles)} profiles from reviewed, attributed events")
    print(json.dumps(report, indent=2))
    if exposure is None:
        print("No exposure minutes supplied: no per-90 rates computed")


def cmd_calibrate(a):
    from .calibration.homography import calibrate, project_tracks
    from .schemas import read_tracks
    t = calibrate(a.landmarks, a.max_median_error_m, a.accept_unvalidated)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "transforms.json").write_text(json.dumps(t, indent=2))
    for s in t["segments"]:
        e = s["median_check_error_m"]
        print(f"frames {s['start_frame']}-{s['end_frame']}: {s['status']} "
              f"(check={s['check_method']}, median error={'n/a' if e is None else f'{e:.2f} m'}) {s['reason']}")
    if a.tracks:
        proj = project_tracks(read_tracks(a.tracks), t)
        proj.to_parquet(out / "tracks_field.parquet", index=False)
        print(f"Projected {int(proj['calibrated'].sum())}/{len(proj)} track rows; others explicitly uncalibrated")


def cmd_identity_validate(a):
    from .identity.assignments import validate_assignments
    from .schemas import read_tracks
    errs = validate_assignments(pd.read_csv(a.assignments, dtype={"player_id": str, "team_id": str, "jersey_evidence": str}),
                                read_tracks(a.tracks), pd.read_csv(a.roster, dtype=str), a.match_date,
                                set(a.team_ids.split(",")) if a.team_ids else None)
    for e in errs:
        print("VIOLATION:", e)
    print("OK" if not errs else f"{len(errs)} violation(s)")
    return 0 if not errs else 1


def cmd_evaluate_events(a):
    from .evaluation.metrics import event_metrics
    from .schemas import read_review_csv
    pred, truth = read_review_csv(a.pred), read_review_csv(a.truth)
    for df in (pred, truth):
        df["video_time_s"] = pd.to_numeric(df["video_time_s"])
    if a.confirmed_only:
        pred = pred[pred["review_status"].isin(["verified", "corrected"])]
    m = event_metrics(pred, truth, a.tolerance)
    if a.output:
        m.to_csv(a.output, index=False)
    print(m.to_string(index=False))


def cmd_process(a):
    from .pipeline.runner import process, summarize
    repo = _repo()
    try:
        state = process(repo, a.match_id, a.video, a.config, a.manifest, a.stop_after, a.force)
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        runs = repo / "data" / "processed" / a.match_id
        print(f"See manifest.json under {runs}", file=sys.stderr)
        return 1
    print(summarize(state).to_string(index=False))
    print(f"\nRun {state['run_id']}: data/processed/{a.match_id}/{state['run_id']}/manifest.json")


def cmd_db_import(a):
    from .db.importer import import_run
    from .db.session import make_engine
    n = import_run(make_engine(), _repo(), Path(a.run_dir))
    print(json.dumps(n, indent=2))


def build_parser():
    p = argparse.ArgumentParser(prog="soccer-vision", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("normalize", help="Normalize a video to constant FPS (original preserved)")
    s.add_argument("--input", required=True); s.add_argument("--output", required=True)
    s.add_argument("--fps", type=int, default=25); s.add_argument("--max-width", type=int, default=1920)
    s.add_argument("--crf", type=int, default=18); s.add_argument("--clip-seconds", type=float)
    s.add_argument("--start-seconds", type=float, default=0.0)
    s.set_defaults(fn=cmd_normalize)

    s = sub.add_parser("validate-manifest")
    s.add_argument("--manifest", required=True); s.add_argument("--production", action="store_true")
    s.add_argument("--check-files", action="store_true")
    s.set_defaults(fn=cmd_validate_manifest)

    s = sub.add_parser("export-detections", help="Step 6 video inference to tracks.csv / balls.csv")
    s.add_argument("--video", required=True); s.add_argument("--match-id", required=True)
    s.add_argument("--player-weights", required=True); s.add_argument("--ball-weights")
    s.add_argument("--class-map", default="config/classmaps/roboflow_football.yaml")
    s.add_argument("--out-dir", required=True); s.add_argument("--device")
    s.add_argument("--fps", type=float, help="default: read from the (constant-FPS) video")
    s.add_argument("--imgsz", type=int, default=1280); s.add_argument("--person-conf", type=float, default=0.25)
    s.add_argument("--ball-conf-keep", type=float, default=0.05); s.add_argument("--max-frames", type=int)
    s.add_argument("--annotated-video"); s.add_argument("--video-id"); s.add_argument("--run-id")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("candidates", help="Step 10 unverified transfer candidates")
    s.add_argument("--match-id", required=True); s.add_argument("--tracks", required=True)
    s.add_argument("--balls", required=True); s.add_argument("--fps", type=float, required=True)
    s.add_argument("--out-dir", required=True)
    s.set_defaults(fn=cmd_candidates)

    s = sub.add_parser("highlights", help="Review clips + clip_index.csv")
    s.add_argument("--video", required=True); s.add_argument("--events", required=True)
    s.add_argument("--out-dir", required=True); s.add_argument("--before", type=float, default=4.0)
    s.add_argument("--after", type=float, default=6.0)
    s.set_defaults(fn=cmd_highlights)

    s = sub.add_parser("profiles", help="Step 14 verified-only player profiles")
    s.add_argument("--events", required=True); s.add_argument("--output", required=True)
    s.add_argument("--exposure", help="CSV match_id,player_id,reviewed_minutes")
    s.set_defaults(fn=cmd_profiles)

    s = sub.add_parser("calibrate", help="Step 7 homographies from landmark correspondences")
    s.add_argument("--landmarks", required=True); s.add_argument("--out-dir", required=True)
    s.add_argument("--tracks"); s.add_argument("--max-median-error-m", type=float, default=2.0)
    s.add_argument("--accept-unvalidated", action="store_true")
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("identity-validate", help="Step 8.6 hard constraints on tracklet assignments")
    s.add_argument("--assignments", required=True); s.add_argument("--tracks", required=True)
    s.add_argument("--roster", required=True); s.add_argument("--match-date")
    s.add_argument("--team-ids", help="comma-separated home,away team IDs")
    s.set_defaults(fn=cmd_identity_validate)

    s = sub.add_parser("evaluate-events", help="Step 18 per-class event metrics vs independent ground truth")
    s.add_argument("--pred", required=True); s.add_argument("--truth", required=True)
    s.add_argument("--tolerance", type=float, default=1.0); s.add_argument("--output")
    s.add_argument("--confirmed-only", action="store_true")
    s.set_defaults(fn=cmd_evaluate_events)

    s = sub.add_parser("process", help="Step 24 orchestrated, resumable run")
    s.add_argument("--match-id", required=True); s.add_argument("--video", required=True)
    s.add_argument("--config", default="config/pipeline.yaml"); s.add_argument("--manifest")
    s.add_argument("--stop-after"); s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_process)

    s = sub.add_parser("db-import-run", help="Load a processing run's metadata and candidates into the database")
    s.add_argument("--run-dir", required=True)
    s.set_defaults(fn=cmd_db_import)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    rc = a.fn(a)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
