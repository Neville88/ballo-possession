"""List a submission's possessions and stoppages with video timestamps, for checking by eye.

    python solution/review.py                                  # reviews ballo
    python solution/review.py evaluation/results/<name>.csv

Writes evaluation/review/<name>_segments.csv, one row per run of the same label:

    start    end      seconds  label
    07:27.11 07:41.03 13.7     H

Timestamps are positions in data/tactical/BAR-ATH_tactical.mp4 (mm:ss.frame at 25 fps, minutes
running on past 60), so any video player can jump straight to them. The file stays under
evaluation/, which is gitignored for CSVs -- it is derived from the match data and must not be
published.

Watching to find where the model is wrong is fine; writing labels from what you see is not. Note
what happened ("quick free kick, play never stopped"), and the fix goes into the algorithm.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAMES, GAME  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
DEFAULT = os.path.join(ROOT, "evaluation", "results", "ballo.csv")
OUT_DIR = os.path.join(ROOT, "evaluation", "review")


def timestamp(frame):
    """mm:ss.ff, minutes running on past 60 (ff = frame within the second)."""
    seconds, sub = divmod(int(frame), FPS)
    minutes, seconds = divmod(seconds, 60)
    return "{:02d}:{:02d}.{:02d}".format(minutes, seconds, sub)


def segments(labels):
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(labels)]))
    return [(s, e, labels[s]) for s, e in zip(starts, ends) if labels[s] != "X"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission", nargs="?", default=DEFAULT)
    args = parser.parse_args()

    with open(args.submission) as handle:
        next(handle)
        labels = np.array([line.rstrip().split(",")[1].strip().upper() for line in handle])

    name = os.path.splitext(os.path.basename(args.submission))[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name + "_segments.csv")
    rows = segments(labels)
    with open(path, "w") as handle:
        handle.write("start,end,seconds,label\n")
        for start, end, label in rows:
            handle.write("{},{},{:.1f},{}\n".format(
                timestamp(start), timestamp(end), (end - start) / FPS, label))
    print("{} segments -> {}".format(len(rows), os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
