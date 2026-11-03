"""Per-frame dead-ball evidence: a naive-Bayes log-likelihood ratio learned from Smart Tagging.

The reference windows (pseudo.py) come from the automatic tagging, never from a ground truth: the
seconds around each set-piece tag start are taken as dead, tagged in-possession phases away from
set pieces as alive. For each signal below, a histogram in each set gives a likelihood ratio per
value; their logs are summed into one score per frame, positive when the frame looks like a
stoppage.

Signals, and what separates them on this match (medians, dead vs alive windows):
  tempo        mean player speed over 2 s              0.97  vs 1.80 m/s
  sprinters    players above 4 m/s, over 2 s           0     vs 1
  ball seen    fraction of the last 4 s with a ball    0.00  vs 1.00
  to line      ball distance to the nearest line       3.2   vs 13.7 m   (ball seen only)
  ball speed   centred-difference speed                2.9   vs 5.4 m/s  (ball seen only)

The signals are correlated, so the summed log ratio overstates the evidence; possession.py scales
it down before use.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from possession import ball_speed, GAMES, GAME, FPS  # noqa: E402

# Params
BINS = 24
SMOOTHING = 1.0             # add-one counts per bin, so an empty bin is not infinite evidence
LLR_CLIP = 3.0              # per signal
TEMPO_WINDOW_S = 2.0
SEEN_WINDOW_S = 4.0
SPRINT_MS = 4.0


def _rolling_mean(values, window):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    kernel = np.ones(int(window))
    total = np.convolve(np.where(valid, values, 0.0), kernel, mode="same")
    count = np.convolve(valid.astype(np.float64), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def signals(data):
    """{name: per-frame array}, NaN where a signal does not exist on that frame."""
    players, ball = data["players"], data["ball"]
    length, width = GAMES[GAME]["pitch"]
    with np.errstate(all="ignore"):
        speed = players[:, :, 2]
        tempo = np.nanmean(speed, axis=1)
        sprinters = np.sum(speed > SPRINT_MS, axis=1).astype(np.float64)
        bx, by = ball[:, 0], ball[:, 1]
        to_line = np.minimum.reduce([bx, length - bx, by, width - by])
    seen = (~np.isnan(ball[:, 0])).astype(np.float64)
    return {
        "tempo": _rolling_mean(tempo, TEMPO_WINDOW_S * FPS),
        "sprinters": _rolling_mean(sprinters, TEMPO_WINDOW_S * FPS),
        "ball seen": _rolling_mean(seen, SEEN_WINDOW_S * FPS),
        "to line": to_line.astype(np.float64),
        "ball speed": ball_speed(ball).astype(np.float64),
    }


class DeadBallModel:
    """Histogram likelihood ratios per signal, fitted on boolean dead / alive masks."""

    def fit(self, sig, dead, alive):
        self.tables = {}
        for name, values in sig.items():
            pool = values[(dead | alive) & np.isfinite(values)]
            if len(pool) < 100:
                continue
            edges = np.unique(np.percentile(pool, np.linspace(0, 100, BINS + 1)))
            edges[0], edges[-1] = -np.inf, np.inf
            hist = []
            for mask in (dead, alive):
                v = values[mask & np.isfinite(values)]
                counts = np.histogram(v, bins=edges)[0] + SMOOTHING
                hist.append(counts / counts.sum())
            llr = np.clip(np.log(hist[0] / hist[1]), -LLR_CLIP, LLR_CLIP)
            self.tables[name] = (edges, llr)
        return self

    def score(self, sig):
        total = np.zeros(len(next(iter(sig.values()))))
        for name, (edges, llr) in self.tables.items():
            values = sig[name]
            ok = np.isfinite(values)
            index = np.clip(np.searchsorted(edges, values[ok], side="right") - 1, 0, len(llr) - 1)
            total[ok] += llr[index]
        return total
