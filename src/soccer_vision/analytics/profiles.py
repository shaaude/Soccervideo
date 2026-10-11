"""Step 14: player profiles from human-confirmed, attributed events only.

* Unreviewed / rejected / unobservable events are excluded.
* Confirmed events must use a valid taxonomy type and outcome; anything else is an error.
* A confirmed actor-level event with a BLANK actor is an unhandled review anomaly -> error.
  Reviewers record a known-but-unidentifiable actor as UNKNOWN; those are excluded from player
  rows and reported as unattributed.
* Team-level types (actor_required: false, e.g. ball_out, goal markers) never enter player rows.
* Player goals come from shots with outcome 'goal' (taxonomy local-v1), so nothing is double counted.
* Rates per 90 are produced only when reviewed exposure minutes are supplied; otherwise None.
"""
from __future__ import annotations

import pandas as pd

from ..events.taxonomy import Taxonomy, load_taxonomy
from ..schemas import UNKNOWN_PLAYER

REQUIRED = {"event_id", "match_id", "event_type", "outcome", "actor_player_id", "review_status"}
COLUMNS_OUT = [
    "player_id", "matches_with_reviewed_events", "total_reviewed_events",
    "pass_attempts_resolved", "completed_passes", "pass_outcome_unknown", "passing_accuracy_pct",
    "crosses", "crosses_completed", "shots", "shots_on_target", "goals",
    "ball_recoveries", "interceptions", "tackles_resolved", "tackles_won",
    "corners_taken", "free_kicks_taken", "throw_ins_taken",
    "reviewed_minutes", "shots_per90", "passes_completed_per90",
]


class ProfileError(ValueError):
    pass


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "") | (s.astype(str).str.lower() == "nan")


def profiles_from_events(events: pd.DataFrame, taxonomy: Taxonomy | None = None,
                         exposure: pd.DataFrame | None = None) -> pd.DataFrame:
    profiles, _ = build_profiles(events, taxonomy, exposure)
    return profiles


def build_profiles(events: pd.DataFrame, taxonomy: Taxonomy | None = None,
                   exposure: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    tax = taxonomy or load_taxonomy()
    missing = REQUIRED.difference(events.columns)
    if missing:
        raise ProfileError(f"Missing event columns: {sorted(missing)}")
    if events["event_id"].duplicated().any():
        raise ProfileError("Duplicate event_id: resolve upstream first")
    ev = events.copy()
    for c in ("event_type", "outcome", "review_status", "actor_player_id", "match_id"):
        ev[c] = ev[c].where(~_blank(ev[c]), "").astype(str).str.strip()

    confirmed = ev[ev["review_status"].isin(tax.confirmed_statuses)].copy()
    problems = []
    for r in confirmed.itertuples(index=False):
        errs = tax.validate_review(r.event_type, r.outcome, r.review_status)
        if errs:
            problems.append(f"{r.event_id}: " + "; ".join(errs))
    if problems:
        raise ProfileError("Invalid confirmed events: " + " | ".join(problems[:20]))

    actor_req = confirmed["event_type"].map(tax.actor_required).astype(bool)
    blank_actor = confirmed["actor_player_id"] == ""
    unhandled = confirmed[actor_req & blank_actor]
    if len(unhandled):
        raise ProfileError(
            f"{len(unhandled)} reviewed events lack actor_player_id ({', '.join(unhandled['event_id'][:10])}). "
            f"Handle explicitly in event review (use {UNKNOWN_PLAYER} if the actor cannot be identified).")
    unknown = confirmed[actor_req & (confirmed["actor_player_id"].str.upper() == UNKNOWN_PLAYER)]
    team_level = confirmed[~actor_req]
    attributed = confirmed[actor_req & ~blank_actor & (confirmed["actor_player_id"].str.upper() != UNKNOWN_PLAYER)]

    minutes = None
    if exposure is not None:
        need = {"match_id", "player_id", "reviewed_minutes"}
        if need.difference(exposure.columns):
            raise ProfileError(f"exposure needs columns {sorted(need)}")
        exp = exposure.astype({"player_id": str})
        if (exp["reviewed_minutes"] < 0).any():
            raise ProfileError("negative reviewed_minutes")
        minutes = exp.groupby("player_id")["reviewed_minutes"].sum()

    rows = []
    for pid, g in attributed.groupby("actor_player_id"):
        t, o = g["event_type"], g["outcome"]
        passes = g[t == "pass"]
        comp = int((passes["outcome"] == "completed").sum())
        inc = int((passes["outcome"] == "incomplete").sum())
        resolved = comp + inc
        shots = g[t == "shot"]
        tackles = g[t == "tackle"]
        mins = float(minutes.get(pid)) if minutes is not None and pid in minutes.index else None
        per90 = (lambda n: round(90.0 * n / mins, 2)) if mins and mins > 0 else (lambda n: None)
        rows.append({
            "player_id": pid,
            "matches_with_reviewed_events": int(g["match_id"].nunique()),
            "total_reviewed_events": len(g),
            "pass_attempts_resolved": resolved,
            "completed_passes": comp,
            "pass_outcome_unknown": len(passes) - resolved,
            "passing_accuracy_pct": round(100.0 * comp / resolved, 1) if resolved else None,
            "crosses": int((t == "cross").sum()),
            "crosses_completed": int(((t == "cross") & (o == "completed")).sum()),
            "shots": len(shots),
            "shots_on_target": int(shots["outcome"].isin(["goal", "saved"]).sum()),
            "goals": int((shots["outcome"] == "goal").sum()),
            "ball_recoveries": int((t == "ball_recovery").sum()),
            "interceptions": int((t == "interception").sum()),
            "tackles_resolved": int(tackles["outcome"].isin(["won", "lost"]).sum()),
            "tackles_won": int((tackles["outcome"] == "won").sum()),
            "corners_taken": int((t == "corner").sum()),
            "free_kicks_taken": int((t == "free_kick").sum()),
            "throw_ins_taken": int((t == "throw_in").sum()),
            "reviewed_minutes": mins,
            "shots_per90": per90(len(shots)),
            "passes_completed_per90": per90(comp),
        })
    report = {
        "taxonomy_version": tax.version,
        "events_in_file": len(ev),
        "excluded_not_confirmed": int(len(ev) - len(confirmed)),
        "status_counts": ev["review_status"].replace("", "blank").value_counts().to_dict(),
        "confirmed": len(confirmed),
        "attributed_to_players": len(attributed),
        "confirmed_unknown_actor": len(unknown),
        "team_level_confirmed": len(team_level),
        "exposure_supplied": minutes is not None,
        "note": "Counts of reviewed events only; not complete-match statistics unless review coverage is complete.",
    }
    return pd.DataFrame(rows, columns=COLUMNS_OUT), report
