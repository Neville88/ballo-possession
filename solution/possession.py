"""Frame-level possession: per-frame evidence, then a switch-penalised Viterbi path.

    python solution/possession.py --name <name>     # writes evaluation/results/<name>.csv
    python evaluation/run_evaluation.py             # scores it next to the baselines

To build the submission itself, use run.py.

Why this shape
--------------
The ranking metric counts whole possessions, so the hard part is not "who is nearest the ball on
this frame" but "when did control really change hands". Each frame contributes a score for each of
the three states D / H / A, and a Viterbi pass picks the single label sequence that maximises the
total score minus a fixed cost for every change of state. A brief touch or a deflection does not
pay for two switches, so it is ignored; an opponent who keeps the ball for a second or more does,
so the possession flips. Frames with no ball keep the current state for free, which is exactly how
the possession rules treat a loose or unseen ball.

Evidence used
-------------
  ball + players   which team's nearest player is close to the ball, down-weighted while the
                   ball is travelling fast (a pass in flight belongs to nobody yet)
  dead ball        a log-likelihood ratio over player tempo, sprinters, ball visibility, ball
                   distance to the lines and ball speed, learned from Smart-Tagging set pieces
                   (deadball.py, pseudo.py)
  situations       ball out of play, ball lost at a line, keeper holds and releases, and the
                   kick that restarts play
  Smart Tagging    phase tags (in-possession / defending) as a weak team prior; set pieces,
                   goals and kickoffs as dead-ball structure

The path is decoded twice, for the restart kick, then tidied of stoppages and spells too short or
too still to be real.

Nothing here is fitted to the public ground truth. Constants are football priors, are learned from
the automatic tagging, or were set after watching where the model failed on the video -- see
TECHNICAL.md.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from baseline import write_submission  # noqa: E402
import load_smart_tagging  # noqa: E402
import features  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
D, H, A = 0, 1, 2
LETTERS = np.array(["D", "H", "A"])

# Ball-control evidence
CONTROL_RADIUS_M = 2.0      # e-folding distance of "a player is on the ball"
TEAM_MARGIN_M = 1.5         # how much nearer one team must be to count as clearly nearer
FLIGHT_SPEED_MS = 4.0       # above this the ball is travelling, not controlled
FLIGHT_DECAY_MS = 6.0
W_BALL = 1.0

# Restart in flight (Law 13, 15, 16). The team taking a free kick, throw-in or goal kick has the
# ball until an opponent controls it. The kick itself scores nothing -- a fast ball belongs to
# nobody -- so without this the path leaves the stoppage straight to whoever stands where the ball
# lands. Applied only around the end of each stoppage of the first-pass path: on every pass in
# open play it also delays real interceptions.
CONTROL_M = 1.5             # a slow ball this close to one team's player is under control
FLIGHT_MAX_S = 3.0          # a long kick is down within this; beyond it the ball is loose
RESTART_WINDOW_S = 2.0      # the restart is struck within this of where the path leaves D
W_FLIGHT = 0.5              # half a controlled frame: who played it is weaker than who has it

# Dead-ball evidence: the summed naive-Bayes log ratio from deadball.py, scaled down because its
# five signals are correlated
W_DEAD_LLR = 0.5

# Team-shape evidence: log-odds from teammodel.py. It is ~97% right on long phases but lags a
# turnover by seconds while the teams reorganise, so it would apply only while the ball is unseen,
# and only as a penalty on the unlikely team -- it must never make "alive" outscore "dead".
# Off: on selfcheck.py it gains ~0.015 tagged-team agreement but costs 3 of the 24 tagged
# turnovers, which is inside the noise either way.
W_SHAPE = 0.0

# Goalkeeper holds. The tracker loses the ball in a keeper's hands, so a catch looks like an
# unseen ball and the previous possession runs on until the restart. When the ball vanishes next
# to a keeper inside his own penalty area, his team gets the unseen frames.
KEEPER_RADIUS_M = 3.0       # ball's projected path to the keeper, over the first second
KEEPER_PROJECT_S = 0.5      # a shot is often lost in flight, metres before the keeper's hands
KEEPER_MIN_GAP_S = 1.0      # shorter gaps are tracking blips, not a held ball
KEEPER_HOLD_S = 25.0        # Law 12 says 6 s, but referees allow longer; set after watching holds
                            # on the video. The ball is live, though slow tempo looks like a stoppage
W_KEEPER = 0.5
BOX_DEPTH_M, BOX_HALF_WIDTH_M = 16.5, 20.16

# Ball out of play (Law 9). The tracker often keeps following the ball off the pitch, and a ball
# wholly over a line is dead whoever is near it. It stays dead until the restart is kicked or
# thrown -- for a goal kick the keeper first carries it back onto the pitch, which must not count
# as play resuming. The margin absorbs tracking noise on balls running along a line.
OUT_MARGIN_M = 1.0
SETTLE_SPEED_MS = 3.0       # the ball has stopped or been picked up (or is unseen) ...
SETTLE_S = 1.0              # ... for this long, so the shot or clearance that took it out is over
RESTART_SPEED_MS = 8.0      # then a kick (goal kick, corner), held for RESTART_FRAMES
RESTART_FRAMES = 3
THROW_SPEED_MS = 3.0        # a throw-in is taken from outside: it restarts when the ball comes
                            # back over the touchline moving, however gently it is thrown
OUT_MAX_S = 45.0            # longest restart we wait for (corners, goal kicks with a substitution)
W_OUT = 4.0                 # a ball off the pitch outweighs the tempo signals, which read
                            # players jogging back into position as live play

# Ball lost at a line. Often the tracker loses the ball just before it goes out and never sees it
# off the pitch. Over the match every such gap lasts 5 s or more and ends mostly in D anyway, but
# the tempo signals take seconds to notice; this marks the stoppage from the moment it vanishes.
LOST_LINE_M = 3.5
LOST_GAP_S = 5.0
W_LOST = 3.0

# Goals. The Smart Tagging goal tag spans the whole attack and the celebrations (on this match it
# opens up to a few minutes before the goal and closes up to ~2 min after), so the ball is dead
# from the last frame inside the tag that it is seen in a penalty area -- the tracker usually loses
# it on the shot -- until the kickoff. (The longest unseen gap would also find the shot, except
# when a drinks break falls inside the tag.)

# Smart Tagging. A set-piece tag opens at the stoppage and play resumes ~6-8 s in (pseudo.py).
# Phase tags overlap -- "Build Up", "Ball in Final Third" and "Players in the Box" can all be open
# at once -- and they close seconds after a turnover, so they count once per frame, by majority,
# never stacked: stacked, they outweighed clear tracking evidence for 10-15 s after turnovers.
W_PHASE = 0.3
W_SETPIECE_DEAD = 1.0
W_SETPIECE_TEAM = 0.5
SETPIECE_DEAD_S = (-4.0, 4.0)       # relative to the tag start
SETPIECE_TEAM_S = (8.0, 20.0)
IN_POSSESSION_CODES = {"build up", "progression", "creating chances", "attacking transition",
                       "ball in final third", "ball in the box", "players in the box", "long ball"}
OUT_OF_POSSESSION_CODES = {"defending in attacking third", "defending in middle third",
                           "defending in defensive third", "defensive transition"}

# Path costs, in units of "one frame of clear evidence"
# Chosen with selfcheck.py, never the public ground truth: a match has a few hundred possessions
# once stoppages split them, and 10 gave ~275 on the tracking alone. Heavier smoothing agrees
# slightly better with the long tagged phases, but that is the per-frame bias EVALUATION.md
# warns about.
COST_SWITCH = 10.0          # H <-> A
COST_DEAD = 15.0            # alive <-> D

# Tidying the path. Fetching the ball and restarting takes longer than MIN_STOP_S, so a shorter
# stoppage with the same team on both sides is tracking noise (a ball drifting just over a line,
# or lost in a crowd). Play that lasts under MIN_SPELL_S between two stoppages is not a possession
# either: it is the shot or clearance that went out, read a moment early. A short stoppage inside
# a set-piece tag's dead window stays: the tag says play did stop, the path just caught little of it.
MIN_STOP_S = 3.0
MIN_SPELL_S = 2.0
# Once dead, the ball stays dead until the restart is taken (Law 13), and a restart is a kick or a
# throw. A spell between two stoppages in which a visible ball is never struck is players waiting.


def ball_speed(ball):
    """Ball speed in m/s from a centred difference; NaN wherever either neighbour is unseen."""
    speed = np.full(len(ball), np.nan, dtype=np.float32)
    step = np.linalg.norm(ball[4:] - ball[:-4], axis=1) * FPS / 4.0
    speed[2:-2] = step
    return speed


def nearest_by_team(players, ball, team):
    """(home, away) distance in metres from the ball to that team's nearest player; inf if none."""
    out = []
    for side in (0, 1):
        xy = players[:, team == side, :2]
        dist = np.linalg.norm(xy - ball[:, None, :], axis=2)
        out.append(np.where(np.isnan(dist), np.inf, dist).min(axis=1))
    return out


def evidence_from_tracking(data):
    """(n_frames, 3) scores for D / H / A from the ball and players."""
    players, ball, team = data["players"], data["ball"], data["team"]
    n = len(ball)
    score = np.zeros((n, 3), dtype=np.float64)

    seen = ~np.isnan(ball[:, 0])
    speed = ball_speed(ball)
    d_home, d_away = nearest_by_team(players, ball, team)

    # Who is on the ball. Weight falls off with distance to the nearest player and with ball
    # speed; the sign says which team is nearer.
    nearest = np.minimum(d_home, d_away)
    flying = np.nan_to_num(np.maximum(speed - FLIGHT_SPEED_MS, 0.0), nan=0.0)
    weight = np.exp(-nearest / CONTROL_RADIUS_M) * np.exp(-flying / FLIGHT_DECAY_MS)
    lean = np.tanh(np.nan_to_num(d_away - d_home, nan=0.0, posinf=10.0, neginf=-10.0)
                   / TEAM_MARGIN_M)
    contribution = np.where(seen & np.isfinite(nearest), W_BALL * weight * lean, 0.0)
    score[:, H] += contribution
    score[:, A] -= contribution
    return score


def evidence_flight(data, where):
    """(n_frames, 3): while the ball travels, credit the team that last controlled it -- on the
    frames of the boolean mask `where` only."""
    players, ball, team = data["players"], data["ball"], data["team"]
    n = len(ball)
    score = np.zeros((n, 3), dtype=np.float64)
    speed = ball_speed(ball)
    d_home, d_away = nearest_by_team(players, ball, team)
    slow = np.nan_to_num(speed, nan=np.inf) < FLIGHT_SPEED_MS
    flying = np.nan_to_num(speed, nan=0.0) > FLIGHT_SPEED_MS
    home = slow & (d_home < CONTROL_M) & (d_away - d_home > TEAM_MARGIN_M)
    away = slow & (d_away < CONTROL_M) & (d_home - d_away > TEAM_MARGIN_M)

    owner, since, max_frames = 0, 0, int(FLIGHT_MAX_S * FPS)
    for t in range(n):
        if home[t] or away[t]:
            owner, since = (H if home[t] else A), t
        elif where[t] and where[since] and flying[t] and owner and t - since <= max_frames:
            score[t, owner] += W_FLIGHT
    return score


def keeper_slots(data):
    """(n_frames, 2) slot index of each team's goalkeeper, and (n_frames, 2) x of its own goal.

    Per half, the keeper is the slot that spends most frames inside its own penalty area."""
    import teammodel

    players, team = data["players"], data["team"]
    length, width = GAMES[GAME]["pitch"]
    direction = teammodel.attack_direction(data)
    slots = np.full((len(players), 2), -1, dtype=np.int64)
    goal_x = np.zeros((len(players), 2))
    for _, first, last in half_video_blocks(GAME):
        home_goal = 0.0 if direction[first] > 0 else length
        for side, own in ((0, home_goal), (1, length - home_goal)):
            block = players[first:last + 1, :, :2]
            with np.errstate(invalid="ignore"):
                inside = ((np.abs(block[:, :, 0] - own) < BOX_DEPTH_M)
                          & (np.abs(block[:, :, 1] - width / 2) < BOX_HALF_WIDTH_M))
            counts = np.where(team == side, inside.sum(axis=0), -1)
            slots[first:last + 1, side] = int(counts.argmax())
            goal_x[first:last + 1, side] = own
    return slots, goal_x


def evidence_keeper(data):
    """(n_frames, 3) scores giving a keeper's team the frames while he holds the ball: the ball
    vanishes at his feet or in his hands inside his own box, or reappears there when he releases
    it. Returns the scores and the number of holds and releases found."""
    players, ball = data["players"], data["ball"]
    length, width = GAMES[GAME]["pitch"]
    n = len(ball)
    score = np.zeros((n, 3))
    slots, goal_x = keeper_slots(data)
    seen = ~np.isnan(ball[:, 0])
    starts = np.flatnonzero(seen[:-1] & ~seen[1:]) + 1
    ends = np.flatnonzero(~seen[:-1] & seen[1:]) + 1
    min_gap, hold, probe = int(KEEPER_MIN_GAP_S * FPS), int(KEEPER_HOLD_S * FPS), FPS
    line = np.minimum.reduce([ball[:, 0], length - ball[:, 0], ball[:, 1], width - ball[:, 1]])
    holds = releases = 0
    for start in starts:
        later = ends[ends > start]
        stop = int(later[0]) if len(later) else n
        if stop - start < min_gap or stop - start > hold:
            # too short is a tracking blip; too long is not a hold -- the ball reappears when the
            # keeper releases it, so a longer gap is a stoppage (e.g. he cleared it out unseen)
            continue
        last = ball[start - 1]
        velocity = (last - ball[start - 5]) * FPS / 4.0 if start >= 5 else np.zeros(2)
        velocity = np.nan_to_num(velocity)
        path = last + np.linspace(0.0, KEEPER_PROJECT_S, 6)[:, None] * velocity
        for side, state in ((0, H), (1, A)):
            own = goal_x[start, side]
            if (abs(last[0] - own) > BOX_DEPTH_M
                    or abs(last[1] - width / 2) > BOX_HALF_WIDTH_M):
                continue
            keeper = players[start - 1:start + probe, slots[start, side], :2]
            with np.errstate(invalid="ignore"):
                gaps = np.linalg.norm(keeper[:, None, :] - path[None, :, :], axis=2)
                near = np.nanmin(gaps) if np.isfinite(gaps).any() else np.inf
            if near <= KEEPER_RADIUS_M:
                score[start:stop, state] += W_KEEPER
                score[start:stop, D] -= W_KEEPER
                holds += 1
                break
        else:
            # The catch itself is often unseen -- the tracker loses a cross or a header in the air
            # -- but the release is not: the ball reappears at his feet or leaves his hands. Not
            # after a ball lost at a line, which is a restart, not a release.
            if stop >= n or line[start - 1] <= LOST_LINE_M:
                continue
            first = ball[stop]
            back = ball[stop + 4] - first if stop + 4 < n else np.zeros(2)
            path = first - np.linspace(0.0, KEEPER_PROJECT_S, 6)[:, None] * np.nan_to_num(back) * FPS / 4.0
            for side, state in ((0, H), (1, A)):
                keeper = players[stop, slots[stop, side], :2]
                own = goal_x[stop, side]
                if (not np.isfinite(keeper).all() or abs(keeper[0] - own) > BOX_DEPTH_M
                        or abs(keeper[1] - width / 2) > BOX_HALF_WIDTH_M):
                    continue
                if np.min(np.linalg.norm(path - keeper, axis=1)) <= KEEPER_RADIUS_M:
                    score[start:stop, state] += W_KEEPER
                    score[start:stop, D] -= W_KEEPER
                    releases += 1
                    break
    return score, holds, releases


def evidence_out(data):
    """(n_frames,) dead-ball score from the ball crossing a line until the restart is played:
    the first kick or throw after the ball has settled."""
    ball = data["ball"]
    length, width = GAMES[GAME]["pitch"]
    n = len(ball)
    speed = ball_speed(ball)
    with np.errstate(invalid="ignore"):
        outside = ((ball[:, 0] < -OUT_MARGIN_M) | (ball[:, 0] > length + OUT_MARGIN_M)
                   | (ball[:, 1] < -OUT_MARGIN_M) | (ball[:, 1] > width + OUT_MARGIN_M))
        over_touchline = (ball[:, 1] < -OUT_MARGIN_M) | (ball[:, 1] > width + OUT_MARGIN_M)
        in_width = (ball[:, 1] >= 0) & (ball[:, 1] <= width)
        fast = np.nan_to_num(speed) > RESTART_SPEED_MS
        calm = ~(np.nan_to_num(speed) > SETTLE_SPEED_MS)          # slow or unseen
        thrown = in_width & (np.nan_to_num(speed) > THROW_SPEED_MS)
    kicked = np.convolve(fast, np.ones(RESTART_FRAMES), "same") >= RESTART_FRAMES
    thrown = np.convolve(thrown, np.ones(RESTART_FRAMES), "same") >= RESTART_FRAMES

    dead = np.zeros(n, dtype=bool)
    longest, settle, t = int(OUT_MAX_S * FPS), int(SETTLE_S * FPS), 0
    for t0 in np.flatnonzero(outside):
        if t0 < t:
            continue
        t, still = t0, 0
        # on a touchline the ball coming back in is the throw itself, so there is nothing to
        # wait out; the in-width test already ignores a clearance still flying off the pitch
        while not over_touchline[t0] and t < n and t - t0 < longest:
            still = still + 1 if calm[t] else 0
            if still >= settle:
                break
            t += 1
        restart = thrown if over_touchline[t0] else kicked
        while t < n and t - t0 < longest and not restart[t]:
            t += 1
        if t - t0 >= longest:
            # no restart seen (often the ball is untracked when it is played): claim only the
            # frames out of play and leave the rest to the dead-ball model
            last_out = t0 + int(np.flatnonzero(outside[t0:t])[-1])
            t = min(t, last_out + settle)
        dead[t0:t] = True
    return np.where(dead, W_OUT, 0.0)


def evidence_lost(data, keeper_score):
    """(n_frames,) dead-ball score over long gaps that start with the ball at a line, unless the
    gap is a keeper hold."""
    ball = data["ball"]
    length, width = GAMES[GAME]["pitch"]
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    line = np.minimum.reduce([ball[:, 0], length - ball[:, 0], ball[:, 1], width - ball[:, 1]])
    held = (keeper_score[:, H] + keeper_score[:, A]) > 0
    out = np.zeros(n)
    starts = np.flatnonzero(seen[:-1] & ~seen[1:]) + 1
    ends = np.flatnonzero(~seen[:-1] & seen[1:]) + 1
    for start in starts:
        later = ends[ends > start]
        stop = int(later[0]) if len(later) else n
        if stop - start >= LOST_GAP_S * FPS and line[start - 1] <= LOST_LINE_M and not held[start]:
            out[start:stop] = W_LOST
    return out


def goal_moments(data, events):
    """Frame of each tagged goal: the last frame inside the tag with the ball in a penalty area."""
    ball = data["ball"]
    length, width = GAMES[GAME]["pitch"]
    with np.errstate(invalid="ignore"):
        in_box = ((np.minimum(ball[:, 0], length - ball[:, 0]) < BOX_DEPTH_M)
                  & (np.abs(ball[:, 1] - width / 2) < BOX_HALF_WIDTH_M))
    moments = []
    for event in events:
        if (event["code"] or "").strip().lower() != "goals":
            continue
        start, end = max(0, event["start_frame"]), min(len(ball), event["end_frame"])
        hits = np.flatnonzero(in_box[start:end])
        moments.append(start + int(hits[-1]) if len(hits) else end)
    return moments


def evidence_dead(data):
    """(n_frames,) dead-ball log ratio, cross-fitted: each half is scored by a model fitted on the
    other half's reference windows, so no frame is scored by a model that saw it."""
    import deadball
    import pseudo

    n = len(data["ball"])
    sig = deadball.signals(data)
    dead, alive_team = pseudo.windows(n)
    alive = alive_team > 0
    out = np.zeros(n)
    blocks = half_video_blocks(GAME)
    for (_, first, last), (_, o_first, o_last) in zip(blocks, blocks[::-1]):
        other = np.zeros(n, dtype=bool)
        other[o_first:o_last + 1] = True
        model = deadball.DeadBallModel().fit(sig, dead & other, alive & other)
        part = {name: values[first:last + 1] for name, values in sig.items()}
        out[first:last + 1] = model.score(part)
    return out


def team_code(name):
    """Smart Tagging club name -> H / A, None for N/A or anything unknown."""
    name = (name or "").strip().lower()
    if name.startswith("barcelona"):
        return H
    if name.startswith("atletic") or name.startswith("athletic"):
        return A
    return None


def evidence_from_tagging(n, use_phases=True, use_setpieces=True, data=None):
    """(n_frames, 3) scores from Smart Tagging, plus a boolean mask of frames forced to D.

    The two switches exist for honest self-checks: pseudo.py builds its reference windows from
    phases and set pieces, so a model being checked against them must not also be fed them."""
    score = np.zeros((n, 3), dtype=np.float64)
    forced_dead = np.zeros(n, dtype=bool)
    votes = np.zeros(n)
    events = load_smart_tagging.load()

    for event in events:
        code = (event["code"] or "").strip().lower()
        side = team_code(event["tags"].get("Team"))
        start, end = max(0, event["start_frame"]), min(n, event["end_frame"])
        if end <= start:
            continue

        if code in IN_POSSESSION_CODES and side is not None and use_phases:
            votes[start:end] += 1.0 if side == H else -1.0
        elif code in OUT_OF_POSSESSION_CODES and side is not None and use_phases:
            votes[start:end] += -1.0 if side == H else 1.0
        elif code == "set pieces" and use_setpieces:
            # Dead around the stoppage the tag opens on; the restarting team has the ball once
            # play resumes.
            lo = max(0, start + int(SETPIECE_DEAD_S[0] * FPS))
            score[lo:start + int(SETPIECE_DEAD_S[1] * FPS), D] += W_SETPIECE_DEAD
            if side is not None:
                score[start + int(SETPIECE_TEAM_S[0] * FPS):
                      min(n, start + int(SETPIECE_TEAM_S[1] * FPS)), side] += W_SETPIECE_TEAM

    score[votes > 0, H] += W_PHASE
    score[votes < 0, A] += W_PHASE

    # Hard structure: from each half's first graded frame up to its tagged kickoff, and from a
    # goal to the kickoff that follows it, the ball is dead.
    kickoffs = sorted(((e["start_frame"] + e["end_frame"]) // 2, (e["tags"].get("Type") or "").lower())
                      for e in events
                      if (e["code"] or "").strip().lower() == "set pieces"
                      and "kick off" in (e["tags"].get("Type") or "").lower())
    for _, first, last in half_video_blocks(GAME):
        starts = [k for k, kind in kickoffs if "start" in kind and first <= k <= last]
        if starts:
            forced_dead[first:min(starts)] = True

    goals = (goal_moments(data, events) if data is not None else
             [e["end_frame"] for e in events if (e["code"] or "").strip().lower() == "goals"])
    for goal in goals:
        after = [k for k, kind in kickoffs if "goal" in kind and k > goal]
        if after and after[0] - goal < 5 * 60 * FPS:
            forced_dead[goal + 1:after[0]] = True
    return score, forced_dead


def viterbi(score, forced_dead):
    """Best D/H/A path through one half: sum of scores minus a cost per change of state."""
    cost = np.array([[0.0, COST_DEAD, COST_DEAD],
                     [COST_DEAD, 0.0, COST_SWITCH],
                     [COST_DEAD, COST_SWITCH, 0.0]])
    n = len(score)
    emit = score.copy()
    emit[forced_dead, H] = -1e9
    emit[forced_dead, A] = -1e9

    back = np.zeros((n, 3), dtype=np.int8)
    total = emit[0].copy()
    for t in range(1, n):
        options = total[:, None] - cost             # from row-state to column-state
        back[t] = options.argmax(axis=0)
        total = options.max(axis=0) + emit[t]

    path = np.empty(n, dtype=np.int8)
    path[-1] = int(total.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def predict(data=None, use_phases=True, use_setpieces=True):
    data = data if data is not None else features.load()
    n = len(data["ball"])
    score = evidence_from_tracking(data)
    score[:, D] += W_DEAD_LLR * evidence_dead(data)
    keeper = evidence_keeper(data)[0]
    score += keeper
    score[:, D] += evidence_out(data)
    score[:, D] += evidence_lost(data, keeper)
    if W_SHAPE:
        import teammodel
        shape = np.where(np.isnan(data["ball"][:, 0]), teammodel.log_odds(data), 0.0)
        score[:, A] -= W_SHAPE * np.maximum(shape, 0.0)
        score[:, H] -= W_SHAPE * np.maximum(-shape, 0.0)
    tag_score, forced_dead = evidence_from_tagging(n, use_phases, use_setpieces, data)
    score += tag_score

    labels = decode(score, forced_dead)
    if W_FLIGHT:
        # Second pass: credit the restart kick out of each stoppage the first pass found.
        ends = np.flatnonzero((labels[:-1] == "D") & np.isin(labels[1:], ["H", "A"])) + 1
        where = np.zeros(n, dtype=bool)
        for e in ends:
            where[max(0, e - int(RESTART_WINDOW_S * FPS)):e + int(FLIGHT_MAX_S * FPS)] = True
        labels = decode(score + evidence_flight(data, where), forced_dead)
    return tidy(labels, forced_dead | (tag_score[:, D] > 0), data["ball"])


def decode(score, forced_dead):
    labels = np.full(len(score), "X", dtype="<U1")
    for _, first, last in half_video_blocks(GAME):
        segment = slice(first, last + 1)
        labels[segment] = LETTERS[viterbi(score[segment], forced_dead[segment])]
    return labels


def tidy(labels, keep_dead, ball):
    """Remove stoppages and spells too short to be real, within each half, and spells between
    stoppages with no sign of the ball being played."""
    labels = labels.copy()
    kick = np.nan_to_num(ball_speed(ball), nan=0.0) > RESTART_SPEED_MS
    for k in range(1, RESTART_FRAMES):
        kick[:-k] &= kick[k:]
    seen = ~np.isnan(ball[:, 0])
    for _, first, last in half_video_blocks(GAME):
        part = labels[first:last + 1]
        changes = np.flatnonzero(part[1:] != part[:-1]) + 1
        starts, ends = np.r_[0, changes], np.r_[changes, len(part)]
        kinds = part[starts].copy()
        for i in range(1, len(starts) - 1):
            lo, hi = first + starts[i], first + ends[i]
            seconds = (hi - lo) / FPS
            if (kinds[i] == "D" and seconds < MIN_STOP_S and kinds[i - 1] == kinds[i + 1]
                    and not keep_dead[lo:hi].any()):
                part[starts[i]:ends[i]] = kinds[i - 1]
            elif kinds[i] != "D" and kinds[i - 1] == kinds[i + 1] == "D":
                no_kick = not kick[lo:hi].any() and seen[lo:hi].mean() > 0.5
                if seconds < MIN_SPELL_S or no_kick:
                    part[starts[i]:ends[i]] = "D"
    return labels


def summarise(labels):
    """Counts and possession statistics -- the submission's own numbers, nothing from a GT."""
    graded = labels[labels != "X"]
    counts = {k: int(np.sum(graded == k)) for k in "HAD"}
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    runs = np.split(labels, changes)
    possessions = [len(r) for r in runs if r[0] in "HA"]
    print("graded frames  H {H}  A {A}  D {D}".format(**counts))
    print("possessions    {}   mean {:.1f} s".format(
        len(possessions), np.mean(possessions) / FPS if possessions else 0.0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="possession", help="submission file name, no extension")
    args = parser.parse_args()

    labels = predict()
    path = write_submission(os.path.join(RESULTS_DIR, args.name + ".csv"), labels)
    summarise(labels)
    print("-> {}".format(os.path.relpath(path, ROOT)))
