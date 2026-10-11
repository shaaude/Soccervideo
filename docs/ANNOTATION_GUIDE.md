# Event annotation guide (taxonomy `local-v1`)

Applies to everyone labelling or reviewing events, and to the code (`config/event_taxonomy.yaml` is the
source of truth; change both together and bump `version`).

## Statuses
| status | meaning |
|---|---|
| `unreviewed` | machine hypothesis, never counted |
| `verified` | reviewer confirms the event as recorded |
| `corrected` | reviewer confirms after changing time, actor or receiver of a model event (set automatically) |
| `rejected` | no such event happened |
| `unobservable` | camera did not show enough to decide; never guess |

`ball_transfer_candidate` is machine-only: choose the real type (pass, cross, interception …) before confirming.

## Actors
* Enter the **roster player ID**, not the tracker number. Tracker IDs are temporary per video run.
* Event certain but player not identifiable → actor `UNKNOWN`. It is excluded from player stats and reported.
* Blank actor on a confirmed actor-level event is an error and blocks profile building.
* Team-level types (`ball_out`, `goal` marker) take no actor.

## Definitions and edge cases
* **Pass** – intentional ball to a teammate or space. Completed if the next *controlled* touch is a teammate's;
  incomplete if an opponent's or the ball leaves play. Deflections that change possession → incomplete.
  Unintentional touches are not passes.
* **Cross** – pass from a wide area into the penalty area; labelled `cross`, not `pass`.
* **Shot** – intentional attempt on goal. Outcomes: goal, saved, blocked, missed. Player goals come only
  from `shot` + `goal`; the separate `goal` event is a team-level marker (prevents double counting, own goals).
* **Ball recovery** – gains control of a loose ball with no clear opponent possession.
* **Interception** – cuts out an opponent's pass before its target.
* **Tackle** – challenge on the ball carrier; won if the tackler's team gains control.
* **Contested possession** with no clear control → no recovery/interception.
* **Goalkeeper** handling is not a pass; distribution by hand or foot is a pass.
* **Assists** are not awarded in `local-v1`. A pass before a goal is not automatically an assist.
* **Restarts** (corner, free kick, throw-in) record the taker as actor.

## Coverage
* Annotate **every** observable event in the reviewed interval, including ones with no model candidate
  (sidebar "Add a missed event"). Without this, recall cannot be measured.
* Record intervals where the camera did not show play as coverage gaps; never label them as zero events.

## Do not mix label sets
SoccerNet action-spotting labels are not these labels. Use an explicit mapping file before combining them.
