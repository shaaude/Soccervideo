# Implementation brief for a GitHub coding agent (next PR series)

Supersedes guide Appendix B, whose PR 1–5 are now implemented (see `IMPLEMENTATION_STATUS.md`).

## Working rules (unchanged in spirit)
1. Read the repository, `docs/IMPLEMENTATION_STATUS.md`, schemas (`src/soccer_vision/schemas.py`) and tests first.
2. One feature branch and one reviewable PR per item below. Never commit footage, weights, crops, embeddings, `.env` or tokens.
3. Keep the four invariants (README). Generated events stay `unreviewed` candidates; identities stay unknown until reviewed.
4. Preserve predictions, corrections, run hashes and model checksums. Never overwrite `original_prediction` / `predicted_*`.
5. No invented metrics. Every reported number comes from labelled ground truth with numerator and denominator.
6. Run `python -m pytest` (and with `SOCCER_TEST_DATABASE_URL` for PostgreSQL). Paste real output. Do not claim tests ran if they did not.

## PR A — Real-weights validation (needs network access to Google Drive)
* Run `scripts/bootstrap_reference.sh`; record weight checksums in `locks/`.
* Run `soccer-vision process` with `config/pipeline.yaml` on the Roboflow demo clip `2e57b9_0.mp4`; attach the stage summary.
* Gate: class-map check passes against the real checkpoints; annotated MP4 inspected by a human.

## PR B — Automatic pitch keypoints (Step 7)
* Wrap the Roboflow pitch-keypoint model (or sn-calibration) to propose landmarks per frame window; feed them
  through `calibration.homography.calibrate` unchanged so the held-out error gate still applies.
* Gate: on frames with surveyed landmarks, median held-out error < 2 m; frames failing stay uncalibrated.

## PR C — Jersey OCR + re-ID evidence (Steps 8, 25)
* Evidence producers write per-tracklet candidates (number, confidence, embedding distance) next to
  `identity_review_queue.csv`; they never write `player_id` directly.
* Calibrate confidences on reviewed crops; do not multiply uncalibrated scores into a "probability".
* Gate: wrong-high-confidence rate (`evaluation.metrics.identity_metrics`) reported per camera type.

## PR D — Results-pipeline integration (Steps 15, 27)
* Import schedules/scores/ratings from the existing results pipeline into `matches`, `external_id_mappings`
  and `team_rating_snapshots` (effective date + input cutoff). Match by provider ID, then teams+date+competition+score;
  collisions → `mapping_status='collision'` for manual review. Never join on team name alone.
* Gate: tests show a post-match rating is never used for a pre-match comparison.

## PR E — Production worker and auth (Steps 16, 29)
* Replace the in-process job thread with Redis + RQ (or Celery) and a separate GPU worker; keep job states.
* Replace API keys with OIDC; add guardian↔player links before enabling the guardian role.
* Object storage with signed URLs; retention/deletion jobs; monitoring of stage latency, GPU time, reviewer minutes.
* Cross-run de-duplication in the database: the file-based profile build excludes confirmed reviews whose
  candidate belongs to a superseded run (`orphaned`), but the API profile query does not yet. Add a
  `superseded_by_run` marker on `events` (migration) and exclude those rows, or serve the newest run's queue only.

## PR F — Temporal event model (Steps 11, 26) — only after the five-match pilot has labels
* Reproduce sn-spotting first. Split by whole matches (70/15/15 + unseen camera/team holdouts).
* Separate heads or stages for presence, type, time, actor track, roster actor, outcome; class-specific thresholds; abstention.
* Integrate as the `temporal_model` stage; outputs remain unreviewed candidates.
* Gate: measured gain over `contact_v2` on held-out whole matches with `evaluate-events`.

## Required final response per PR
Files changed, commands run, test results (real stdout summary), dependency changes, limitations, example outputs,
and the next highest-priority failure mode.
