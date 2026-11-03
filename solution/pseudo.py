"""Ground-truth-free reference windows, built from the automatic Smart Tagging.

The public five minutes may not be used for tuning (RULES.md), so parameters need another yardstick.
Smart Tagging is automatic -- not human labels, not derived from the ground truth -- and it covers
the whole match. Two kinds of window come out of it:

  dead    the seconds around the start of each set-piece tag. A set-piece tag is ~35 s long and
          opens at the stoppage, not the restart: averaged over all 59, player tempo bottoms out
          and the ball goes unseen from ~4 s before the tag start to ~4 s after it, and play
          resumes ~6-8 s in
  alive   frames inside a team's in-possession phase, away from any set piece; the tagged team
          is the one with the ball

Both are noisy (the tagging has false positives and its timing is approximate), so they are a
yardstick for comparing variants of a model over the full match, not labels to copy.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, HERE)
import load_smart_tagging  # noqa: E402
from possession import (FPS, H, A, IN_POSSESSION_CODES, OUT_OF_POSSESSION_CODES,  # noqa: E402
                        team_code, half_video_blocks, GAME)

# Params
DEAD_AROUND_S = (-3.0, 3.0)     # dead window, relative to the set-piece tag start
SETPIECE_GUARD_S = 8.0          # alive windows keep this far from any set piece
TRANSITION_CHECK_S = (2.0, 6.0) # after an attacking-transition tag starts, the tagged team has it


def _graded(n):
    graded = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    return graded


def _near_setpiece(events, n):
    near = np.zeros(n, dtype=bool)
    guard = int(SETPIECE_GUARD_S * FPS)
    for event in events:
        if (event["code"] or "").strip().lower() == "set pieces":
            near[max(0, event["start_frame"] - guard):event["end_frame"] + guard] = True
    return near


def team_windows(n):
    """(n_frames,) H / A where the tagging says which team has the ball, 0 elsewhere.

    In-possession phases vote for their team and defending phases for the other one. Frames with
    votes for both, and frames near a set piece, are left out."""
    events = load_smart_tagging.load()
    votes = np.zeros((n, 3), dtype=np.int16)
    for event in events:
        code = (event["code"] or "").strip().lower()
        side = team_code(event["tags"].get("Team"))
        if side is None:
            continue
        lo, hi = max(0, event["start_frame"]), min(n, event["end_frame"])
        if code in IN_POSSESSION_CODES:
            votes[lo:hi, side] += 1
        elif code in OUT_OF_POSSESSION_CODES:
            votes[lo:hi, H + A - side] += 1
    out = np.where(votes[:, H] > 0, H, 0) + np.where(votes[:, A] > 0, A, 0)
    out[(votes[:, H] > 0) & (votes[:, A] > 0)] = 0
    out[_near_setpiece(events, n) | ~_graded(n)] = 0
    return out.astype(np.int8)


def transition_hits(labels):
    """Fraction of attacking-transition tags after which the model gives the ball to the tagged
    team -- a check on switch timing, which per-frame agreement barely sees."""
    hits = []
    for event in load_smart_tagging.load():
        if (event["code"] or "").strip().lower() != "attacking transition":
            continue
        side = team_code(event["tags"].get("Team"))
        if side is None:
            continue
        lo = event["start_frame"] + int(TRANSITION_CHECK_S[0] * FPS)
        hi = event["start_frame"] + int(TRANSITION_CHECK_S[1] * FPS)
        window = labels[lo:hi]
        hits.append(np.mean(window == ("H" if side == H else "A")) > 0.5)
    return float(np.mean(hits)) if hits else float("nan")


def windows(n):
    """(dead_mask, alive_team) -- alive_team is H / A on alive frames, 0 elsewhere."""
    events = load_smart_tagging.load()
    dead = np.zeros(n, dtype=bool)
    near_setpiece = np.zeros(n, dtype=bool)
    alive_team = np.zeros(n, dtype=np.int8)

    graded = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True

    for event in events:
        if (event["code"] or "").strip().lower() != "set pieces":
            continue
        lo = max(0, event["start_frame"] + int(DEAD_AROUND_S[0] * FPS))
        hi = max(0, event["start_frame"] + int(DEAD_AROUND_S[1] * FPS))
        dead[lo:hi] = True
        guard = int(SETPIECE_GUARD_S * FPS)
        near_setpiece[max(0, event["start_frame"] - guard):event["end_frame"] + guard] = True

    for event in events:
        code = (event["code"] or "").strip().lower()
        side = team_code(event["tags"].get("Team"))
        if code in IN_POSSESSION_CODES and side is not None:
            alive_team[max(0, event["start_frame"]):event["end_frame"]] = side
    alive_team[near_setpiece] = 0

    return dead & graded, np.where(graded, alive_team, 0)


def report(labels, dead, alive_team, name=""):
    """Agreement of a label array with the reference windows. Higher is better on all three."""
    dead_hit = np.mean(labels[dead] == "D")
    alive = alive_team > 0
    alive_hit = np.mean(labels[alive] != "D")
    team_letters = np.where(alive_team == H, "H", "A")
    team_hit = np.mean(labels[alive] == team_letters[alive])
    turnover_hit = transition_hits(labels)
    print("{:<22} dead-window D {:.3f}   alive-window alive {:.3f}   tagged-team {:.3f}   "
          "turnovers {:.3f}".format(name, dead_hit, alive_hit, team_hit, turnover_hit))
    return dead_hit, alive_hit, team_hit, turnover_hit
