"""Try different values for one number in brain.WEIGHTS against an opponent brain.

Usage:
    python tune.py KEY VALUE1 VALUE2 ... [--games N] [--opponent NAME]

Example:
    python tune.py territory 5 10 15 --games 200 --opponent old_python/brain_v4

For each value, brain.WEIGHTS[KEY] is set in memory (brain.py is not edited),
then N games are played against the opponent (default old_python/brain_v3) with the game
engine from simulate.py. Both snakes think for only TUNE_TIME_LIMIT_MS per move,
and games run in parallel on all CPU cores.
"""

import math
import multiprocessing
import os
import random
import sys

import simulate

DEFAULT_GAMES = 100
DEFAULT_OPPONENT = "old_python/brain_v3"
TUNE_TIME_LIMIT_MS = 20

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so the brains (and brain_v2 fallback) can be imported

_brains = {}  # filled once per worker process


def load_brains(opponent):
    """Import brain and the opponent once per process, with the short time limit."""
    if not _brains:
        import brain

        brain.TIME_LIMIT_MS = TUNE_TIME_LIMIT_MS
        # The opponent is loaded as its own copy, so changing brain.WEIGHTS never affects it.
        opp_move = simulate.load_brain(opponent)
        if "TIME_LIMIT_MS" in opp_move.__globals__:
            opp_move.__globals__["TIME_LIMIT_MS"] = TUNE_TIME_LIMIT_MS
        _brains["brain"] = brain
        _brains["opponent"] = opp_move
        _brains["original"] = dict(brain.WEIGHTS)
    return _brains


def play_one(task):
    """Play one game with WEIGHTS[key] = value. Returns (value, 'win'/'loss'/'draw')."""
    key, value, game, base_seed, opponent = task
    b = load_brains(opponent)
    brain = b["brain"]
    brain.WEIGHTS.clear()
    brain.WEIGHTS.update(b["original"])
    brain.WEIGHTS[key] = value

    # Alternate starting sides; each pair of games shares the same start layout.
    # Every value plays the exact same layouts, so the comparison is fair.
    if game % 2 == 0:
        brains, names, me = [brain.choose_move, b["opponent"]], ["brain", opponent], 0
    else:
        brains, names, me = [b["opponent"], brain.choose_move], [opponent, "brain"], 1
    setup_rng = random.Random(base_seed + game // 2)
    rng = random.Random(base_seed * 7919 + game)
    winner = simulate.play_game(f"tune-{key}-{value}-{game}", brains, names, setup_rng, rng, {})

    if winner is None:
        return value, "draw"
    return value, "win" if winner == me else "loss"


def wilson(wins, n, z=1.96):
    """95% confidence range for a win rate (Wilson score interval, works at 0% and 100%)."""
    p = wins / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - margin, center + margin


def parse_number(text):
    try:
        return int(text)
    except ValueError:
        return float(text)


def read_option(args, flag, default):
    """Remove '--flag VALUE' from args and return VALUE (or the default)."""
    if flag in args:
        i = args.index(flag)
        value = args[i + 1]
        del args[i:i + 2]
        return value
    return default


def main():
    args = sys.argv[1:]
    games = int(read_option(args, "--games", DEFAULT_GAMES))
    opponent = read_option(args, "--opponent", DEFAULT_OPPONENT)
    if len(args) < 2:
        sys.exit("Usage: python tune.py KEY VALUE1 VALUE2 ... [--games N] [--opponent NAME]")

    import brain

    simulate.load_brain(opponent)  # stops here with a clear message if the file is missing
    key = args[0]
    values = list(dict.fromkeys(parse_number(v) for v in args[1:]))  # drop duplicates
    if key not in brain.WEIGHTS:
        sys.exit(f"Unknown key {key!r}. Choose one of: {', '.join(brain.WEIGHTS)}")

    print(f"Tuning WEIGHTS[{key!r}] (current value {brain.WEIGHTS[key]}) against {opponent}")
    print(f"{games} games per value, {TUNE_TIME_LIMIT_MS} ms per move, "
          f"{os.cpu_count()} processes\n")

    base_seed = random.randrange(10**9)
    tasks = [(key, v, g, base_seed, opponent) for v in values for g in range(games)]
    results = {v: {"win": 0, "loss": 0, "draw": 0} for v in values}

    with multiprocessing.Pool(os.cpu_count()) as pool:
        for done, (value, outcome) in enumerate(pool.imap_unordered(play_one, tasks), 1):
            results[value][outcome] += 1
            print(f"\r  {done}/{len(tasks)} games done", end="", flush=True)
    print("\n")

    print(f"{'value':>10} {'wins':>6} {'losses':>7} {'draws':>6} {'win rate':>9}"
          f"   95% confidence")
    for v in values:
        r = results[v]
        low, high = wilson(r["win"], games)
        print(f"{v:>10} {r['win']:>6} {r['loss']:>7} {r['draw']:>6} "
              f"{100 * r['win'] / games:>8.1f}%   {100 * low:.1f}% to {100 * high:.1f}%")


if __name__ == "__main__":
    main()
