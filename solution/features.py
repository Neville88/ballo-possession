"""Parse the tactical ATD once into numpy arrays, and cache them next to the data.

The raw file is ~190 MB of text and takes a minute or two to stream; the cache loads in about a
second. It lives under data/cache/, which is gitignored like the rest of data/ -- the arrays are
the tracking data itself, so they must never be committed.

    load()  ->  {
        "players": (n_frames, n_slots, 3) float32   x, y in metres, speed in m/s; NaN = not seen
        "ball":    (n_frames, 2) float32            x, y in metres; NaN = not seen
        "team":    (n_slots,) int8                  0 home, 1 away
    }

Index 0 of every array is video frame 0, i.e. submission row 0 (ATD frame_id - 1).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import DATA_DIR, GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402

# Params
CACHE_PATH = os.path.join(DATA_DIR, "cache", "tactical_atd.npz")
CACHE_VERSION = 1


def _parse_group(text, width):
    """'x,y,s' -> floats, NaN-padded to `width`. Tolerates an empty group."""
    values = [float(v) for v in text.split(",") if v != ""][:width]
    return values + [np.nan] * (width - len(values))


def parse_atd(raw_path, n_frames, n_slots):
    """Stream the raw ATD into arrays in normalised 0-1 units. See load() for the layout."""
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue

            groups = parts[1].split(";")
            if len(groups) == n_slots:
                try:
                    flat = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)
                    players[frame] = flat.reshape(n_slots, -1)[:, :3]
                except ValueError:
                    players[frame] = [_parse_group(g, 3) for g in groups]
            else:
                for slot, group in enumerate(groups[:n_slots]):
                    players[frame, slot] = _parse_group(group, 3)

            ball[frame] = _parse_group(parts[2], 2)
    return players, ball


def load(game=GAME, data_dir=None, rebuild=False):
    """The tactical ATD as arrays in metres. Parses on first call, then reads the cache."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team = team_of_each_track(meta_path)

    if not rebuild and os.path.isfile(CACHE_PATH):
        cached = np.load(CACHE_PATH)
        if int(cached["version"]) == CACHE_VERSION:
            return {"players": cached["players"], "ball": cached["ball"], "team": team}

    print("parsing {} (one-off, a minute or two) ...".format(os.path.basename(raw_path)))
    players, ball = parse_atd(raw_path, source_frames("tactical", game), len(team))

    # Normalised 0-1 -> metres. The pitch is not square, so distances need this before anything
    # compares them.
    length, width = GAMES[game]["pitch"]
    players[..., 0] *= length
    players[..., 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez(CACHE_PATH, players=players, ball=ball, version=CACHE_VERSION)
    return {"players": players, "ball": ball, "team": team}
