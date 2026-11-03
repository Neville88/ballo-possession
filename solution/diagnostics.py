"""Which tracking signals separate dead ball from live play? Summary percentiles only.

    python solution/diagnostics.py

Compares each candidate per-frame signal inside the Smart-Tagging dead windows against the alive
windows (see pseudo.py). Prints percentiles, never frames -- the output is safe to share.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features  # noqa: E402
import pseudo  # noqa: E402
from possession import ball_speed, nearest_by_team, GAMES, GAME  # noqa: E402

# Params
PERCENTILES = (10, 25, 50, 75, 90)


def signals(data):
    players, ball, team = data["players"], data["ball"], data["team"]
    length, width = GAMES[GAME]["pitch"]
    with np.errstate(all="ignore"):
        speed = players[:, :, 2]
        tempo = np.nanmean(speed, axis=1)
        fast = np.sum(speed > 4.0, axis=1).astype(np.float32)
        tracked = np.sum(~np.isnan(players[:, :, 0]), axis=1).astype(np.float32)
        spread_x = np.nanstd(players[:, :, 0], axis=1)
        d_home, d_away = nearest_by_team(players, ball, team)
        nearest = np.minimum(d_home, d_away)
        nearest[~np.isfinite(nearest)] = np.nan
        bx, by = ball[:, 0], ball[:, 1]
        edge = np.minimum.reduce([bx, length - bx, by, width - by])
    seen = (~np.isnan(ball[:, 0])).astype(np.float32)
    # Fraction of the surrounding 4 s in which the ball is seen.
    k = 4 * 25
    seen_frac = np.convolve(seen, np.ones(k) / k, mode="same")
    # Tempo smoothed over 2 s.
    tempo_s = np.convolve(np.nan_to_num(tempo), np.ones(50) / 50, mode="same")
    return {
        "player tempo m/s": tempo,
        "tempo 2s avg m/s": tempo_s,
        "players > 4 m/s": fast,
        "tracked players": tracked,
        "x spread m": spread_x,
        "ball seen (4s frac)": seen_frac,
        "ball speed m/s": ball_speed(ball),
        "nearest to ball m": nearest,
        "ball to line m": edge,
    }


def main():
    data = features.load()
    dead, alive_team = pseudo.windows(len(data["ball"]))
    alive = alive_team > 0
    print("dead-window frames {}   alive-window frames {}\n".format(dead.sum(), alive.sum()))
    print("{:<22} {:<6} ".format("signal", "set") + " ".join("p{:<5}".format(p) for p in PERCENTILES))
    for name, values in signals(data).items():
        for label, mask in (("dead", dead), ("alive", alive)):
            v = values[mask]
            v = v[np.isfinite(v)]
            cells = " ".join("{:<6.2f}".format(x) for x in np.percentile(v, PERCENTILES)) if len(v) else "n/a"
            print("{:<22} {:<6} {}   (n={})".format(name if label == "dead" else "", label, cells, len(v)))


if __name__ == "__main__":
    main()
