# Implementation status against the guide (Steps 1–30)

Legend: **Built** = implemented and covered by tests. **Built, unvalidated** = code exists and runs, but
needs real footage or labels to judge quality. **Needs you** = requires footage, permissions, people,
GPU time or decisions this repository cannot supply. **Not started** = deliberately out of scope for
this increment.

| Step | Topic | Status | Where / what is missing |
|---|---|---|---|
| 1 | Bounded first deliverable | **Needs you** | Choose the authorized 10–15 min elevated wide-angle segment; record rights and timing in the manifest. |
| 2 | Environment | Built | `pyproject.toml`, extras per stage; `locks/working-environment.txt`. GPU wheel choice is yours. |
| 3 | Roboflow reference | Built (pinning, headless patch); **needs you** to download weights | `scripts/bootstrap_reference.sh`, `locks/roboflow-sports-commit.txt`, `docs/patches/`. Google Drive was blocked in the build environment, so the 5 demo videos were not rendered here. |
| 3.3 | Normalize own video | Built | `soccer-vision normalize` (original preserved, provenance JSON, CFR check). |
| 3.3+ | Video from a link (Veo) | Built | `soccer-vision veo` / `ingest`, `POST /matches/{id}/videos/from-url`: allowlisted https hosts, private-address and redirect checks, resumable download, token redaction, dual-feed manifest, follow-cam review clips after a verified offset, tiled detection and polynomial calibration for the stitched panorama. Not tested against Veo's servers (blocked from the build environment); tested against a local server with the same URL layout. See `docs/VEO.md`. |
| 4 | Repository structure | Built | As guide; `tracking` lives in `detection/export.py` (ByteTrack). |
| 5 | Manifest + provenance | Built | `config/match001.json` template; `validate-manifest --production` refuses unresolved nulls / non-granted permission. |
| 6 | Detection export | Built | Detector-agnostic class maps validated against `model.names`; schema validation; ranked ball alternatives; annotated MP4; empty-schema output. |
| 7 | Calibration | Built, unvalidated | RANSAC homographies, held-out / leave-one-out error, rejection gate, per-segment frames, foot-point projection, airborne ball not projected. **Needs you:** surveyed landmarks per camera position. Automatic pitch-keypoint detection (Roboflow pitch model / sn-calibration) is not integrated. |
| 8 | Team and identity | Partial | Built: team-colour clustering (needs reviewed cluster→team mapping), sharpest crops, assignment template, hard-constraint validator, identity metrics. Not built: jersey OCR, re-ID embeddings, sn-gamestate integration (needs labelled crops and a fine-tuned model). |
| 9 | Taxonomy + annotation rules | Built | `config/event_taxonomy.yaml`, `docs/ANNOTATION_GUIDE.md`. |
| 10 | Contact/transfer candidates + clips | Built | `contact_v2`: enforces the 0.12–4.0 s gap, first-touch timing, suppression reasons; clips validated before cutting and probed after. |
| 11 | Temporal event-and-actor model | **Not started** | Needs SoccerNet access approval, annotated local footage and GPU training. The pipeline stage is present and records `SKIPPED` with the reason. |
| 12 | Human review | Built | Streamlit app: event type, outcome, corrected time, match clock, actor/receiver track + roster ID, UNKNOWN, add missed events, optimistic locking, audit history, merge of re-generated candidates. CVAT: use upstream as documented in the guide. |
| 13 | PostgreSQL schema | Built | 16 tables + Alembic migration, CHECK constraints (including "candidate cannot be confirmed"), indexes; tested on PostgreSQL 16 (`compose.yaml` uses 17). Frame-level data stays in CSV/Parquet per run. |
| 14 | Verified-only profiles | Built | UNKNOWN / team-level handling, outcome validation, single-counted goals, per-90 only with reviewed exposure, coverage report. |
| 15 | League results integration | Schema built; **needs you** | `external_id_mappings`, `team_rating_snapshots` (effective date + input cutoff). Connecting your existing results/Elo pipeline needs its data export; no join on team names. |
| 16 | FastAPI + jobs | Built (pilot) | All 9 guide endpoints plus add-missed-event and signed media. Job executor is an in-process thread; production should use Redis+RQ/Celery with a separate GPU worker. |
| 17 | Next.js frontend | **Not started** | Streamlit covers the local pilot. |
| 18 | Evaluation suite | Built (metrics); **needs labels** | Per-class precision/recall/F1 with numerators and denominators, timing error, joint event+actor F1, detection recall, IDF1/MOTA via motmetrics, identity abstention / wrong-high-confidence rate. |
| 19 | Model-learning loop | **Not started** | Requires a trained model and MLflow; reviewed corrections are already stored as candidate labels with full history. |
| 20 | PR organization | See `docs/AGENT_BRIEF.md` | |
| 21 | Architecture / headless | Built | Separate stores for observations, hypotheses and reviews; headless patch; run-scoped IDs; explicit paths and FPS. |
| 22 | Kit integration | Built | All kit scripts replaced by tested package modules; guide CLIs kept in `scripts/`. |
| 23 | Five-match pilot | **Needs you** | Permissions, footage, an independent annotator. Use `evaluate-events` to compute misses as well as false alarms. |
| 24 | Orchestrated command | Built | `soccer-vision process`: run hash, per-stage status, checksums, elapsed time, resume, fail-fast, explicit SKIPPED reasons. |
| 25 | Longitudinal identity service | Partial | Dated rosters + constraint checks + metrics built. Probabilistic assignment from OCR/re-ID evidence not built. |
| 26 | Temporal classification upgrade | **Not started** | Same dependency as Step 11. |
| 27 | Canonical match linking | Schema built | Mapping workflow (provider ID → teams/date/score → manual collision review) still to implement against your sources. |
| 28 | Longitudinal profiles | Partial | Verified profiles + clips via API. Context-adjusted metrics need a validated statistical model; natural-language query not built. |
| 29 | Private pilot deployment | Partial | Role checks at backend, signed expiring media, access audit table, no public binding. Not built: OIDC, object storage, retention/deletion jobs, monitoring. |
| 30 | Agent handoff | Built | `docs/AGENT_BRIEF.md` (updated for remaining work). |

## Changes from the guide's kit (with reasons)

* **Minimum transfer gap enforced.** The guide specifies 0.12–4.0 s; kit code only had the maximum.
* **First-touch timing.** Kit episodes started after the 3-frame support window, so receiver times were
  ~2 frames late. Episodes now start at the first supporting contact.
* **Suppression reasons kept** (`suppressed_candidates.csv`), per Step 10.5.
* **Player IDs stay strings** (kit read `12` back as `12.0`).
* **Team-level events and UNKNOWN actors** no longer block profile building; a blank actor on a confirmed
  actor-level event still does, as the kit intended.
* **Goals counted once**, from `shot` + outcome `goal`; `goal` events are team-level markers. The kit's
  review app (`goal`) and tests (`scored`) used different outcome words; the taxonomy now fixes the vocabulary.
* **Clips:** all timestamps validated before any clip is cut (the kit could leave a partial set without an
  index); event IDs are sanitized for filenames; every clip is probed after encoding.
* **Review app:** paths are arguments instead of hard-coded `match001`; adds missed events, time
  correction, receiver, concurrency protection; re-generated candidates merge without losing reviews.
* **Kit test adjusted:** the kit's synthetic "pass" teleported the ball between feet in one frame (0.04 s),
  which the guide's minimum gap correctly rejects. The test now includes ball flight, and a separate test
  checks that the teleport is suppressed with reason `gap_below_min`.
* **supervision pinned below 0.31:** `sv.ByteTrack` is deprecated since 0.28 and removed in 0.31.
