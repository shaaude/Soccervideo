# Soccer Vision

Evidence-linked soccer video analytics. The system detects and tracks players and the ball, proposes
**candidate** events, routes them to human review, and builds player profiles only from
**confirmed** events. It implements the *Soccer Vision — Complete Open-Source Implementation Guide*
(11 October 2026). See [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md) for a
step-by-step account of what is built and what still needs footage, labels or a GPU.

Four invariants are enforced in code, not only in documentation:

1. Observations are not verified events: generated events are `ball_transfer_candidate` / `unreviewed`.
   The taxonomy, review store, database constraint and API all refuse to confirm that type.
2. Tracking IDs are not roster identities: track IDs are scoped to `(video_id, run_id, segment_id)`.
   Roster identities come only from human-reviewed assignments.
3. Missing observations are not zeros: uncalibrated frames have `NaN` field coordinates; skipped
   stages are recorded as `SKIPPED` with a reason; profiles report coverage and exclusions.
4. No unreviewed prediction appears as a confirmed statistic: aggregation reads only `verified` /
   `corrected` events with a resolved actor.

## Layout

```
app/review.py                  local-only Streamlit reviewer (Step 12)
config/                        match manifests, pipeline configs, class maps, event taxonomy
db/migrations/                 Alembic migrations (PostgreSQL schema, Step 13)
docs/                          status, annotation guide, agent brief, patches, examples
scripts/                       guide-compatible CLIs (thin wrappers over `soccer-vision`)
src/soccer_vision/
  detection/   class maps + video exporter (detector + ByteTrack -> tracks.csv / balls.csv)
  calibration/ landmark homographies, held-out error, foot-point projection
  identity/    team-appearance clustering, review crops, assignment constraint checks
  events/      taxonomy, contact/transfer candidates, review clips
  review/      file-backed review store with optimistic locking + append-only audit
  analytics/   verified-only player profiles
  evaluation/  per-class event, detection, identity and tracking metrics
  pipeline/    orchestrated, resumable runner with run hashes and manifests (Step 24)
  db/ api/     SQLAlchemy models, run importer, FastAPI service (Step 16)
tests/                         63 tests (unit, pipeline, API on SQLite + PostgreSQL, migrations, Streamlit app)
```

## Set up

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# PyTorch: pick the wheel for your hardware at https://pytorch.org/get-started/locally/
pip install -e ".[vision,db,api,review,eval,dev]"
sudo apt install -y ffmpeg
bash scripts/bootstrap_reference.sh      # Roboflow Sports at the pinned commit + demo weights (Google Drive)
export SOCCER_DEVICE=cuda                # or cpu / mps
python -m pytest                         # 63 tests; PostgreSQL tests need SOCCER_TEST_DATABASE_URL
```

Licensing: the Roboflow example models depend on Ultralytics (AGPL-3.0). Review before distributing.

## Analyse a Veo recording from its links

```bash
soccer-vision veo --match-id <match> --permission granted --process \
  --panorama "https://c.veocdn.com/<recording>/panorama/transcode-<id>.mp4" \
  --standard "https://c.veocdn.com/<recording>/standard/machine/<id>/video.mp4"
```
The panorama is analysed and the follow-cam is used for review clips. See [`docs/VEO.md`](docs/VEO.md)
for the offset check, panorama calibration, and GPU needs.

## Process a match from a local file

```bash
# 1. Preserve the original and normalize to constant FPS
cp /path/to/match.mp4 data/raw/match001-original.mp4
soccer-vision normalize --input data/raw/match001-original.mp4 --output data/normalized/match001.mp4 --fps 25

# 2. Fill in config/match001.json from verified sources (teams, date, pitch size, permission)
soccer-vision validate-manifest --manifest config/match001.json --production

# 3. Run every stage; resumable, provenance in data/processed/match001/<run_hash>/manifest.json
soccer-vision process --match-id match001 --video data/normalized/match001.mp4 --config config/pipeline.yaml

# 4. Review candidates locally (bind to localhost only)
streamlit run app/review.py --server.address 127.0.0.1 -- --match-id match001 \
  --video data/normalized/match001.mp4 \
  --candidates data/processed/match001/<run_hash>/event_candidates.csv \
  --clips data/processed/match001/<run_hash>/highlights/clip_index.csv

# 5. Re-run: unchanged stages are reused; aggregation builds verified-only profiles
soccer-vision process --match-id match001 --video data/normalized/match001.mp4 --config config/pipeline.yaml
```

Each stage also runs on its own (`soccer-vision export-detections | candidates | highlights | calibrate |
identity-validate | profiles | evaluate-events`), and the guide's script names in `scripts/` accept the
guide's flags.

### Field calibration
Add `"calibration_landmarks": "config/calibration/match001.json"` to the manifest. The format is in
`docs/calibration_landmarks.example.json`: at least 4 non-collinear landmarks per camera segment,
surveyed pitch length and width, and at least one `holdout` point. Segments with median held-out error
above `calibration.max_median_error_m` (default 2 m) are rejected, and their frames stay uncalibrated.

### Identities
The `identify` stage writes team-colour clusters (unmapped cluster indices, not teams), the five
sharpest crops per tracklet, and an empty `tracklet_assignments_template.csv`. Reviewers fill it in.
`soccer-vision identity-validate` checks it against the roster (dated memberships, jersey evidence,
no player in two tracklets at once).

## Database and API

```bash
echo "POSTGRES_PASSWORD=$(openssl rand -hex 24)" > .env && docker compose up -d db
export SOCCER_DATABASE_URL=postgresql+psycopg://soccer:<password>@127.0.0.1:5432/soccervision
alembic upgrade head
soccer-vision db-import-run --run-dir data/processed/match001/<run_hash> --with-reviews
SOCCER_API_KEYS="<key>:admin:alice,<key2>:reviewer:bob" SOCCER_MEDIA_SECRET=$(openssl rand -hex 32) \
  uvicorn soccer_vision.api.app:app --host 127.0.0.1
```

Endpoints follow guide Step 16. Roles: admin, coach, reviewer, analyst. Coaches and analysts never see
unreviewed hypotheses. Guardian access is denied until a verified guardian-player link exists. Media
links are HMAC-signed and expire. API keys are a pilot mechanism; use your identity provider before
exposing anything beyond localhost.

## Verified in this repository

`docs/SMOKE_TEST.md` records an end-to-end CPU run on a 10.7-second public broadcast clip using a
community SoccerNet-trained YOLOv8 checkpoint (the Roboflow weights were unreachable from the build
environment). The run shows the pipeline works mechanically. It is **not** an accuracy measurement:
accuracy claims require the five-match pilot with independent annotation (Step 23).
