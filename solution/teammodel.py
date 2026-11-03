"""Which team has the ball, from the shape of both teams -- usable when the ball is not seen.

The ball is missing on 41% of graded frames, but all 22 players usually are not. A team in
possession stretches the pitch and pushes up; the team without it stays compact, sits between the
ball and its own goal and runs back. This module learns that from the automatic Smart Tagging:
in-possession phases name the team with the ball, defending phases the team without it (both
agree with the nearest player to the ball ~85% of the time on this match, see pseudo.py).

A logistic regression on direction-normalised shape features, cross-fitted by half: each half is
scored by a model trained on the other half's tags. numpy only.

    log_odds(data)  ->  (n_frames,) log-odds that the home team has the ball
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from possession import GAMES, GAME, FPS, H, A, nearest_by_team, half_video_blocks  # noqa: E402

# Params
VELOCITY_LAG = 12               # frames either side for player velocity (~1 s total)
L2 = 1.0                        # ridge penalty on standardised features
NEWTON_STEPS = 25
LOG_ODDS_CLIP = 4.0


def attack_direction(data):
    """(n_frames,) +1 where the home team attacks toward x = pitch length, -1 where toward 0.

    Over a whole half each team's average position sits nearer its own goal, so the team with the
    smaller mean x defends the x = 0 goal. Computed per half, since the teams change ends.
    """
    players, team = data["players"], data["team"]
    direction = np.zeros(len(players))
    with np.errstate(all="ignore"):
        for _, first, last in half_video_blocks(GAME):
            block = players[first:last + 1, :, 0]
            home = np.nanmean(block[:, team == 0])
            away = np.nanmean(block[:, team == 1])
            direction[first:last + 1] = 1.0 if home < away else -1.0
    return direction


def features(data):
    """(n_frames, k) shape features, every x measured in the home team's attacking direction."""
    players, ball, team = data["players"], data["ball"], data["team"]
    length, _ = GAMES[GAME]["pitch"]
    direction = attack_direction(data)
    x = (players[:, :, 0] - length / 2) * direction[:, None]
    y = players[:, :, 1]

    velocity = np.full_like(x, np.nan)
    lag = VELOCITY_LAG
    velocity[lag:-lag] = (x[2 * lag:] - x[:-2 * lag]) * FPS / (2 * lag)

    columns = []
    with np.errstate(all="ignore"):
        for side in (0, 1):
            mask = team == side
            columns += [
                np.nanmean(x[:, mask], axis=1),             # how far up the pitch
                np.nanstd(x[:, mask], axis=1),              # length of the block
                np.nanstd(y[:, mask], axis=1),              # width of the block
                np.nanmean(velocity[:, mask], axis=1),      # moving forward or back
                np.nanmax(x[:, mask], axis=1),              # most advanced player
            ]
        d_home, d_away = nearest_by_team(players, ball, team)
        nearest = np.minimum(d_home, d_away)
        lean = np.tanh(np.clip(d_away - d_home, -10, 10) / 1.5) * np.exp(-nearest / 2.0)
        ball_x = (ball[:, 0] - length / 2) * direction
    seen = np.isfinite(lean)
    columns += [np.where(seen, lean, 0.0), np.where(seen, ball_x, 0.0), seen.astype(float)]
    out = np.column_stack(columns).astype(np.float64)
    out[~np.isfinite(out)] = 0.0
    return out


def fit_logistic(X, y):
    """Ridge logistic regression by Newton's method. Returns (mean, scale, weights)."""
    mean, scale = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mean) / scale])
    w = np.zeros(Z.shape[1])
    penalty = np.full(Z.shape[1], L2)
    penalty[0] = 0.0
    for _ in range(NEWTON_STEPS):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        gradient = Z.T @ (p - y) + penalty * w
        hessian = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(penalty)
        w -= np.linalg.solve(hessian, gradient)
    return mean, scale, w


def predict_log_odds(model, X):
    mean, scale, w = model
    Z = np.column_stack([np.ones(len(X)), (X - mean) / scale])
    return np.clip(Z @ w, -LOG_ODDS_CLIP, LOG_ODDS_CLIP)


def log_odds(data):
    """(n_frames,) cross-fitted log-odds that the home team has the ball; 0 outside the halves."""
    import pseudo

    X = features(data)
    labels = pseudo.team_windows(len(X))
    out = np.zeros(len(X))
    blocks = half_video_blocks(GAME)
    for (_, first, last), (_, o_first, o_last) in zip(blocks, blocks[::-1]):
        train = np.zeros(len(X), dtype=bool)
        train[o_first:o_last + 1] = True
        train &= labels > 0
        model = fit_logistic(X[train], (labels[train] == H).astype(float))
        out[first:last + 1] = predict_log_odds(model, X[first:last + 1])
    return out
