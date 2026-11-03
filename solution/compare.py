"""Compare any number of submissions over the full match, without a ground truth.

    python solution/compare.py                          # every CSV in evaluation/results/
    python solution/compare.py a.csv b.csv c.csv        # specific files, e.g. each teammate's

Run it next to the organisers' scorer, which covers only the public five minutes:

    python evaluation/run_evaluation.py

The two answer different questions. The public window is 17 possessions -- the README shows it
choosing the wrong baseline -- so it says whether a pipeline is correct, not which one is better.
This script looks at all 106 minutes against the automatic Smart Tagging (see pseudo.py):

  dead       share of tagged stoppages labelled D
  alive      share of tagged open play not labelled D
  team       share of tagged open play given to the tagged team
  turnovers  share of the 24 tagged ball recoveries after which the recovering team has it
  poss/dur   possessions over the full match and their mean length; a match has a few hundred

Caveat: a submission built from the Smart Tagging scores higher here partly because it read the
same tags. Compare like with like, and treat a gap smaller than a few points as a tie.
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import possession  # noqa: E402
import pseudo  # noqa: E402

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
N_FRAMES = 199319


def read(path):
    with open(path) as handle:
        next(handle)
        labels = np.array([line.rstrip().split(",")[1].strip().upper() for line in handle])
    if len(labels) != N_FRAMES:
        raise ValueError("{}: {} rows, expected {}".format(path, len(labels), N_FRAMES))
    return labels


def stats(labels):
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    lengths = [len(r) for r in np.split(labels, changes) if r[0] in "HA"]
    return len(lengths), (np.mean(lengths) / possession.FPS if lengths else 0.0)


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(RESULTS_DIR, "*.csv")))
    dead, alive_team = pseudo.windows(N_FRAMES)
    alive = alive_team > 0
    team_letters = np.where(alive_team == possession.H, "H", "A")

    print("{:<26} {:>6} {:>6} {:>6} {:>9} {:>6} {:>6}".format(
        "submission", "dead", "alive", "team", "turnovers", "poss", "dur"))
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            labels = read(path)
        except (ValueError, OSError, IndexError) as error:
            print("{:<26} skipped: {}".format(name, error))
            continue
        count, mean_s = stats(labels)
        print("{:<26} {:>6.3f} {:>6.3f} {:>6.3f} {:>9.3f} {:>6d} {:>6.1f}".format(
            name,
            np.mean(labels[dead] == "D"),
            np.mean(labels[alive] != "D"),
            np.mean(labels[alive] == team_letters[alive]),
            pseudo.transition_hits(labels),
            count, mean_s))


if __name__ == "__main__":
    main()
