"""Regenerate the submission from the provided data, in one command.

    pip install -r solution/requirements.txt
    python solution/run.py --team <team_name>

Writes evaluation/results/<team_name>.csv, runs the organisers' validator on it, and prints its
SHA-256 so two runs can be compared byte for byte -- the rules require the repository to
reproduce the submitted file exactly. --check runs the whole pipeline twice from a cold cache and
fails loudly if the two files differ.
"""

import argparse
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import features  # noqa: E402
import possession  # noqa: E402
from baseline import write_submission  # noqa: E402

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
VALIDATOR = os.path.join(ROOT, "evaluation", "validate_submission.py")


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def build(path, rebuild_cache=False):
    data = features.load(rebuild=rebuild_cache)
    labels = possession.predict(data)
    write_submission(path, labels)
    possession.summarise(labels)
    return sha256(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", required=True, help="team name; the CSV is named after it")
    parser.add_argument("--check", action="store_true",
                        help="build twice, the first time re-parsing the raw tracking, and "
                             "confirm the two files are identical")
    args = parser.parse_args()

    path = os.path.join(RESULTS_DIR, args.team + ".csv")
    digest = build(path, rebuild_cache=args.check)
    print("sha256 {}".format(digest))

    if args.check:
        again = build(path)
        if again != digest:
            sys.exit("NOT REPRODUCIBLE: second build gave sha256 {}".format(again))
        print("reproducible: two builds, one from the raw tracking, are byte-identical")

    result = subprocess.run([sys.executable, VALIDATOR, path])
    if result.returncode != 0:
        sys.exit("validation failed -- do not submit this file")
    print("-> {}".format(os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
