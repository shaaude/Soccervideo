#!/usr/bin/env python3
"""LOCAL-ONLY event review app (Step 12). Do not expose as a public service.

    streamlit run app/review.py -- --match-id match001 \
        --video data/normalized/match001-test.mp4 \
        --candidates data/processed/match001/event_candidates.csv \
        --clips data/highlights/match001/clip_index.csv

Defaults match the guide's match001 paths. Bind to localhost only (streamlit's
--server.address 127.0.0.1); production review needs authentication, authorization,
database transactions and signed media URLs (see the FastAPI service).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soccer_vision.events.taxonomy import load_taxonomy  # noqa: E402
from soccer_vision.review.store import ConflictError, ReviewStore, ReviewValidationError  # noqa: E402
from soccer_vision.schemas import UNKNOWN_PLAYER  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--match-id", default="match001")
p.add_argument("--video", default=None)
p.add_argument("--candidates", default=None)
p.add_argument("--annotations-dir", default=None)
p.add_argument("--clips", default=None, help="clip_index.csv from create_highlights")
args, _ = p.parse_known_args(sys.argv[1:])
mid = args.match_id
video = Path(args.video or f"data/normalized/{mid}-test.mp4")
candidates = Path(args.candidates or f"data/processed/{mid}/event_candidates.csv")
ann_dir = Path(args.annotations_dir or f"data/annotations/{mid}")
clips_path = Path(args.clips or f"data/highlights/{mid}/clip_index.csv")

st.set_page_config(page_title="Soccer event review", layout="wide")
st.title(f"Soccer event review — {mid} (local pilot)")
st.caption("Candidates are unverified hypotheses, NOT match statistics. Only verified/corrected events "
           "with a reviewed actor ever reach player profiles.")

tax = load_taxonomy()
store = ReviewStore(ann_dir, mid, tax)
if not video.exists():
    st.error(f"Missing video {video}. Run the pipeline first.")
    st.stop()
if candidates.exists():
    sync = store.sync_candidates(pd.read_csv(candidates, dtype={"event_id": str, "match_id": str}))
    if sync["orphaned"]:
        st.warning(f"{sync['orphaned']} reviewed events no longer appear in the latest candidates (kept, flagged orphaned).")
rows = store.load()
clips = pd.read_csv(clips_path, dtype={"event_id": str}) if clips_path.exists() else pd.DataFrame(columns=["event_id"])

with st.sidebar:
    st.subheader("Queue")
    status_filter = st.multiselect("Status", list(tax.review_statuses), default=["unreviewed"])
    view = rows[rows["review_status"].isin(status_filter)] if status_filter else rows
    counts = rows["review_status"].value_counts().to_dict()
    st.write(counts or "no events")
    st.divider()
    st.subheader("Add a missed event")
    with st.form("add"):
        a_type = st.selectbox("Event type", [t for t in tax.types if not tax.machine_only(t)], key="a_type")
        a_out = st.selectbox("Outcome", tax.outcomes(a_type), key="a_out")
        a_time = st.number_input("Video time (s)", min_value=0.0, step=0.04, key="a_time")
        a_actor = st.text_input(f"Actor roster player ID (or {UNKNOWN_PLAYER})", key="a_actor")
        a_recv = st.text_input("Receiver roster player ID", key="a_recv")
        a_rev = st.text_input("Reviewer ID", key="a_rev")
        if st.form_submit_button("Add event"):
            try:
                eid = store.add_missed_event({"event_type": a_type, "outcome": a_out, "video_time_s": a_time,
                                              "actor_player_id": a_actor, "receiver_player_id": a_recv}, a_rev)
                st.success(f"Added {eid}")
                st.rerun()
            except (ReviewValidationError, ValueError) as e:
                st.error(str(e))

if rows.empty:
    st.info("No events. Zero candidates is not evidence of zero events: check detection coverage "
            "(frames with a ball candidate in detection_meta.json) before interpreting.")
    st.stop()
if view.empty:
    st.info("Nothing in the selected status filter.")
    st.stop()

event_id = st.selectbox("Event", view["event_id"].tolist(),
                        format_func=lambda e: f"{e}  @ {float(rows.loc[rows.event_id == e, 'video_time_s'].iloc[0]):.2f}s")
r = rows[rows["event_id"] == event_id].iloc[0]
left, right = st.columns([3, 2])
with left:
    clip = clips[clips["event_id"] == event_id]
    if len(clip) and Path(clip.iloc[0]["clip_path"]).exists():
        st.video(str(clip.iloc[0]["clip_path"]))
        st.caption(f"Review clip {clip.iloc[0]['start_seconds']}–{clip.iloc[0]['end_seconds']} s of source; "
                   f"event at {float(r['video_time_s']):.2f} s")
    else:
        st.video(str(video), start_time=max(0, int(float(r["video_time_s"]) - 4)))
with right:
    st.json({"source": r["source"], "predicted_type": r["predicted_event_type"],
             "predicted_time_s": r["predicted_video_time_s"],
             "initiator_track": r["predicted_initiator_track_id"], "receiver_track": r["predicted_receiver_track_id"],
             "rule_version": r["rule_version"], "version": int(r["version"]), "orphaned": r["orphaned"] == "true"})

types = tax.types


def idx(opts, value, fallback=0):
    return opts.index(value) if value in opts else fallback


ev_type = st.selectbox("Observed event type", types, index=idx(types, r["event_type"]))
outs = tax.outcomes(ev_type)
with st.form("review"):
    outcome = st.selectbox("Outcome", outs, index=idx(outs, r["outcome"]))
    c1, c2 = st.columns(2)
    vt = c1.number_input("Corrected video time (s)", value=float(r["video_time_s"]), min_value=0.0, step=0.04)
    mc = c2.text_input("Match clock (s), if known", value=str(r["match_clock_s"]))
    c3, c4 = st.columns(2)
    actor_track = c3.text_input("Actor track ID", value=str(r["actor_track_id"]))
    actor = c4.text_input(f"Actor roster player ID (blank = unresolved; {UNKNOWN_PLAYER} = not identifiable)",
                          value=str(r["actor_player_id"]))
    c5, c6 = st.columns(2)
    recv_track = c5.text_input("Receiver track ID", value=str(r["receiver_track_id"]))
    recv = c6.text_input("Receiver roster player ID", value=str(r["receiver_player_id"]))
    state = st.selectbox("Review status", list(tax.review_statuses), index=idx(list(tax.review_statuses), r["review_status"]))
    notes = st.text_area("Reviewer notes", value=str(r["review_notes"]))
    reviewer = st.text_input("Reviewer ID", value=str(r["reviewer_id"]) or "")
    submit = st.form_submit_button("Save review", type="primary")

if submit:
    try:
        store.save_review(event_id, {"event_type": ev_type, "outcome": outcome, "video_time_s": vt,
                                     "match_clock_s": mc, "actor_track_id": actor_track, "actor_player_id": actor,
                                     "receiver_track_id": recv_track, "receiver_player_id": recv,
                                     "review_status": state, "review_notes": notes},
                          reviewer, expected_version=int(r["version"]))
        st.success("Saved review + append-only audit record.")
        st.rerun()
    except ConflictError as e:
        st.error(str(e))
    except (ReviewValidationError, ValueError) as e:
        st.error(str(e))

with st.expander("Audit history for this event"):
    st.write(store.audit_history(event_id))
