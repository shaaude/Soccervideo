# Analysing a Veo recording from its links

Veo gives two files for each recording. Use both:

| Veo feed | Example URL | Used for | Why |
|---|---|---|---|
| **Panorama** (raw stitched ~180° view) | `https://c.veocdn.com/<recording>/panorama/transcode-<id>.mp4` | **Analysis**: detection, tracking, candidates, pitch calibration | The camera never moves and every player is in frame all match, so tracks stay continuous and one calibration covers the whole game. |
| **Standard** (AI follow-cam) | `https://c.veocdn.com/<recording>/standard/machine/<id>/video.mp4` | **Review clips** | Easier for people to watch. It pans and zooms, so it is not used for tracking or field coordinates. |

## One command

```bash
soccer-vision veo --match-id 2026-10-04-gu12-vs-alexandria \
  --panorama "https://c.veocdn.com/<recording>/panorama/transcode-<id>.mp4" \
  --standard "https://c.veocdn.com/<recording>/standard/machine/<id>/video.mp4" \
  --permission granted --process
```

This downloads both files (resumable, size and disk checks, originals kept in `data/raw/`). It normalizes
them to constant 25 FPS. The panorama keeps up to 4096 px width because far-side players are small.
It writes `config/<match>.json` with a `feeds` section and runs the pipeline with
`config/pipeline.veo.yaml`, which turns on tiled detection for the wide panorama.

* Trial on a few minutes first: add `--clip-seconds 300`.
* Without `--process`, it only downloads and prepares; run `soccer-vision process ...` later.
* Two links from different recordings are refused before anything is downloaded.
* Query strings (signed-URL tokens) are never stored; manifests keep the URL without them.
* Only `c.veocdn.com` is accepted by default. Add hosts with `SOCCER_ALLOWED_VIDEO_HOSTS=host1,host2`.

The API equivalent (admin role) is `POST /api/v1/matches/{match_id}/videos/from-url` with
`{"url": ..., "feed": "panorama"|"standard", "video_use_permission": "granted"}`. It returns an ingest
job; the downloaded feed is registered as video `<match>-<feed>` and can be processed with `/process`.

## Before results mean anything: three checks people must make

1. **Permission.** Only process recordings you are allowed to use (your own team's Veo account,
   league/club policy, parents' consent for youth footage). `--permission granted` is your statement of
   that and is stored in the manifest. Videos and crops never go into Git (`data/` is ignored).
2. **Follow-cam time offset.** The two feeds usually line up, but this is not assumed. Compare frames:
   ```bash
   soccer-vision feed-frames --match-id <m> --time 600 --offset 0 --out reports/offset-check.png
   ```
   When the panorama and follow-cam show the same moment, record it:
   ```bash
   soccer-vision feed-offset --match-id <m> --seconds 0 --verified-by "S"
   ```
   Until then, review clips come from the panorama only. After it is set, re-running `process` cuts
   follow-cam clips too (only the clip stage re-runs), and the review app shows both views:
   ```bash
   streamlit run app/review.py --server.address 127.0.0.1 -- --match-id <m> \
     --video data/normalized/<m>-panorama.mp4 \
     --candidates data/processed/<m>/<run>/event_candidates.csv \
     --clips data/processed/<m>/<run>/highlights/clip_index.csv \
     --review-clips data/processed/<m>/<run>/highlights_review/clip_index.csv
   ```
3. **Pitch calibration (once per camera position).** The panorama is stitched, so straight pitch
   lines appear curved and a single homography does not fit well. Use the polynomial model:
   ```bash
   soccer-vision landmark-frame --video data/normalized/<m>-panorama.mp4 --time 120 --out reports/grid.png
   ```
   Read pixel positions of at least 14 pitch landmarks off the gridded frame: corners, halfway-line ends,
   centre spot, penalty-area corners, goal-area corners, penalty spots. Spread them across the whole pitch
   and mark 2–3 as `"holdout": true`. Write them to `config/calibration/<m>.json` with
   `"model": "poly3"`, one segment covering the whole match, and the measured pitch length and width
   (see `docs/calibration_landmarks.example.json`). Then set `"calibration_landmarks"` in the match manifest.
   Segments whose held-out error exceeds 2 m are rejected and their frames stay uncalibrated.
   If the tripod was moved at half-time, use two segments.

## Compute

A full 90-minute panorama is about 135,000 frames, each split into 3–4 tiles. That needs a CUDA GPU
(`SOCCER_DEVICE=cuda`). On a CPU, expect several seconds per frame: use `--clip-seconds` trials only.

## Known limits

* The follow-cam feed is not analysed. Tracking through its pans and zooms would need per-frame
  automatic pitch keypoints (see `AGENT_BRIEF.md`, PR B).
* The Roboflow detectors were trained on broadcast footage. Youth fields, a high side-line camera and
  panorama distortion may lower recall. Measure this on your first pilot match before trusting counts.
* Veo links can expire. A 401/403/404/410 is reported as "link may have expired or need sign-in";
  get a fresh link from Veo and re-run. Completed downloads are reused.
