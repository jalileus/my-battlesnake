"""Try different values for one number in brain.WEIGHTS against brain_v3.

Usage:
    python tune.py KEY VALUE1 VALUE2 ... [--games N]

Example:
    python tune.py territory 6 10 14 --games 100

For each value, brain.WEIGHTS[KEY] is set in memory (brain.py is not edited),
then N games are played against brain_v3 with the game engine from simulate.py.
Both snakes think for only TUNE_TIME_LIMIT_MS per move, and games run in
parallel on all CPU cores.
"""

import multiprocessing
import os
import random
import sys

import simulate

DEFAULT_GAMES = 100
TUNE_TIME_LIMIT_MS = 20

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so the brains (and brain_v2 fallback) can be imported

_brains = {}  # filled once per worker process


def load_brains():
    """Import brain and brain_v3 once per process, with the short time limit."""
    if not _brains:
        import brain
        import brain_v3

        brain.TIME_LIMIT_MS = TUNE_TIME_LIMIT_MS
        brain_v3.TIME_LIMIT_MS = TUNE_TIME_LIMIT_MS
        _brains["brain"] = brain
        _brains["v3"] = brain_v3
        _brains["original"] = dict(brain.WEIGHTS)
    return _brains


def play_one(task):
    """Play one game with WEIGHTS[key] = value. Returns (value, 'win'/'loss'/'draw')."""
    key, value, game, base_seed = task
    b = load_brains()
    brain = b["brain"]
    brain.WEIGHTS.clear()
    brain.WEIGHTS.update(b["original"])
    brain.WEIGHTS[key] = value

    # Alternate starting sides; each pair of games shares the same start layout.
    # Every value plays the exact same layouts, so the comparison is fair.
    if game % 2 == 0:
        brains, names, me = [brain.choose_move, b["v3"].choose_move], ["brain", "brain_v3"], 0
    else:
        brains, names, me = [b["v3"].choose_move, brain.choose_move], ["brain_v3", "brain"], 1
    setup_rng = random.Random(base_seed + game // 2)
    rng = random.Random(base_seed * 7919 + game)
    winner = simulate.play_game(f"tune-{key}-{value}-{game}", brains, names, setup_rng, rng, {})

    if winner is None:
        return value, "draw"
    return value, "win" if winner == me else "loss"


def parse_number(text):
    try:
        return int(text)
    except ValueError:
        return float(text)


def main():
    args = sys.argv[1:]
    games = DEFAULT_GAMES
    if "--games" in args:
        i = args.index("--games")
        games = int(args[i + 1])
        del args[i:i + 2]
    if len(args) < 2:
        sys.exit("Usage: python tune.py KEY VALUE1 VALUE2 ... [--games N]")

    import brain

    key = args[0]
    values = list(dict.fromkeys(parse_number(v) for v in args[1:]))  # drop duplicates
    if key not in brain.WEIGHTS:
        sys.exit(f"Unknown key {key!r}. Choose one of: {', '.join(brain.WEIGHTS)}")

    print(f"Tuning WEIGHTS[{key!r}] (current value {brain.WEIGHTS[key]}) against brain_v3")
    print(f"{games} games per value, {TUNE_TIME_LIMIT_MS} ms per move, "
          f"{os.cpu_count()} processes\n")

    base_seed = random.randrange(10**9)
    tasks = [(key, v, g, base_seed) for v in values for g in range(games)]
    results = {v: {"win": 0, "loss": 0, "draw": 0} for v in values}

    with multiprocessing.Pool(os.cpu_count()) as pool:
        for done, (value, outcome) in enumerate(pool.imap_unordered(play_one, tasks), 1):
            results[value][outcome] += 1
            print(f"\r  {done}/{len(tasks)} games done", end="", flush=True)
    print("\n")

    print(f"{'value':>10} {'wins':>6} {'losses':>7} {'draws':>6} {'win rate':>9}")
    for v in values:
        r = results[v]
        print(f"{v:>10} {r['win']:>6} {r['loss']:>7} {r['draw']:>6} "
              f"{100 * r['win'] / games:>8.1f}%")


if __name__ == "__main__":
    main()
