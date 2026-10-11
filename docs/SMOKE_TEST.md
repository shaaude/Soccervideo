# End-to-end smoke test (11 Oct 2026, CPU-only build environment)

Purpose: show that every stage runs mechanically from raw video to reviewed, gated profiles.
**This is not an accuracy result.** Accuracy needs the five-match pilot with independent annotation (Step 23).

## Inputs
| item | value |
|---|---|
| Video | 10.7 s public broadcast clip (`test_videos/CityUtdR.mp4` in github.com/Mostafa-Nafie/Football-Object-Detection), 1280×720, 25 FPS, 268 frames. Adult professional match, not youth footage. Not committed. |
| Detector | community YOLOv8 checkpoint trained on SoccerNet frames (`weights/best.pt`, same repository, no license file; SoccerNet terms apply). Smoke test only. |
| Why not Roboflow weights | Google Drive (where `setup.sh` downloads them) was unreachable from the build environment. `config/pipeline.yaml` is set up for them. |
| Class map | `config/classmaps/soccernet_yolov8_smoke.yaml` (Player 0, GoalKeeper 1, Ball 2, Main/Side Referee 3/4, Staff 5 dropped) |
| Config | `config/pipeline.smoke.yaml`; manifest `config/smoke001.json` (no calibration landmarks) |
| Code | commit `4c89b4b`; run hash `ea76ed8c288622e4` |

## Stage results (`soccer-vision process`)
| stage | status | records | time | note |
|---|---|---|---|---|
| validate | SUCCEEDED | 1 | 0.05 s | warnings: permission pending; teams/date/pitch unresolved |
| detect_track | SUCCEEDED | 3,356 track rows | 403 s (CPU) | 32 track IDs; ball candidate in 247/268 frames (658 ball rows incl. low-confidence alternatives) |
| calibrate | SKIPPED | – | – | no landmarks: field coordinates explicitly missing |
| identify | SUCCEEDED | 32 tracklets | 2.5 s | 25 player tracklets clustered by kit colour; 5 crops per tracklet; empty assignment template |
| candidates | SUCCEEDED | 3 | 0.3 s | 224 ball frames above threshold; 3 suppressed (2 same-player regains, 1 gap 0.08 s < 0.12 s) |
| temporal_model | SKIPPED | – | – | no trained model |
| highlights | SUCCEEDED | 3 clips | 19 s | each clip probed after encoding |
| review_queue | SUCCEEDED | 3 | – | 3 awaiting review |
| aggregate | SKIPPED → SUCCEEDED | 0 player rows | – | skipped until a review existed; then 1 confirmed event with actor UNKNOWN, 2 rejected |

A second identical run reused every completed stage (detector not re-run). The run was repeated from a
clean directory: identical counts (3,356 rows, 32 IDs, 3 candidates), so the detector + tracker path was
deterministic on this CPU build.

## Observations a reviewer would care about
* Kit-colour clustering split light-blue and red kits correctly on visually checked frames. The match
  referee (black kit) was labelled "Player" by this detector and fell into the red cluster with a low
  margin (0.29 vs 0.6–0.9 for players). This is why clusters require review before team attribution.
* Candidate 1 (track 15 → track 11, 2.4 s gap) was a long ball from a light-blue player collected by a
  red player: a real but **incomplete** pass, which the candidate rule cannot tell apart from a completed one.
* Candidates 2–3 (0.2 s gaps between tracks 11 and 2) were a contested ball near the touchline, so they
  were rejected. Short-gap alternations between two nearby players are a likely false-positive pattern for `contact_v2`.

The three demonstration reviews were entered from a 4-frame contact sheet under reviewer ID
`claude-smoke-demo` and say so in their notes. They exercise the gates and are not authoritative labels.

## Database / API path
`alembic upgrade head` on PostgreSQL 16, then `soccer-vision db-import-run --with-reviews` (twice:
second run changed nothing). Live `uvicorn` checks: admin saw 3 events; coach saw `[]` (no unreviewed
hypotheses); no key → 401.
