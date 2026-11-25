"""Record brain.py's move and positions-searched count at a fixed depth, so the JS
port can be checked against it (tools/verify.js).

Usage (from the repo root): python js/tools/dump_expected.py
Writes js/test/expected.json. Needs pypy_test/data/positions.json (saved positions).
"""
import json
import multiprocessing
import os
import sys

JS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(JS)
DATA = os.path.join(REPO, "pypy_test", "data", "positions.json")


def init():
    sys.path.insert(0, REPO)
    global brain
    import brain
    brain.TIME_LIMIT_MS = 10**9


def run(task):
    i, state, depth = task
    brain.MAX_DEPTH = depth
    move, done, positions = brain.search_move(state)
    return [i, depth, move, positions]


if __name__ == "__main__":
    states = json.load(open(DATA, encoding="utf-8"))
    tasks = [(i, s, 4) for i, s in enumerate(states)] + [(i, states[i], 6) for i in range(0, len(states), 5)]
    with multiprocessing.Pool(6, initializer=init) as pool:
        rows = pool.map(run, tasks, chunksize=20)
    json.dump(rows, open(os.path.join(JS, "test", "expected.json"), "w"))
    print(f"wrote {len(rows)} expected results")
