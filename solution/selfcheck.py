"""Compare model variants over the full match without any ground truth.

    python solution/selfcheck.py

Each variant is scored against the Smart-Tagging reference windows (pseudo.py) with the tag
evidence switched off, so the model is never checked against windows it was fed. Also prints the
possession count and mean duration, which a football match bounds independently of any labels.
This is the yardstick for choosing between versions; the public five minutes are only a pipeline
check.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features  # noqa: E402
import possession  # noqa: E402
import pseudo  # noqa: E402


def possession_stats(labels):
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    lengths = [len(r) for r in np.split(labels, changes) if r[0] in "HA"]
    return len(lengths), np.mean(lengths) / possession.FPS


def main():
    data = features.load()
    dead, alive_team = pseudo.windows(len(data["ball"]))
    labels = possession.predict(data, use_phases=False, use_setpieces=False)
    pseudo.report(labels, dead, alive_team, "tracking only")
    count, mean_s = possession_stats(labels)
    print("{:<22} possessions {}   mean {:.1f} s".format("", count, mean_s))


if __name__ == "__main__":
    main()
