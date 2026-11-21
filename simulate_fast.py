"""Parallel version of simulate.py: plays many games at once on several CPU cores.

Usage:
    python simulate_fast.py brain_a brain_b N [--time MS] [--time-a MS] [--time-b MS] [--procs P]

--time MS     thinking time limit for both snakes (default 100, the real limit)
--time-a MS   time limit for brain_a only (default: --time)
--time-b MS   time limit for brain_b only (default: --time)
--procs P     number of processes (default: CPU cores minus 2)

The game engine is imported from simulate.py, so the rules are exactly the same.
"""

import math
import multiprocessing
import os
import random
import sys
import time

import simulate

DEFAULT_TIME_MS = 100

_brains = []  # (label, choose_move) pairs, loaded once per worker process


def init_worker(name_a, name_b, label_a, label_b, time_a, time_b):
    """Runs once in every worker: load both brains and give each its own time limit."""
    sys.path.insert(0, simulate.HERE)  # so brains can import helper files (e.g. brain_v2)
    for name, label, time_ms in ((name_a, label_a, time_a), (name_b, label_b, time_b)):
        choose_move = simulate.load_brain(name)
        # Each brain is its own module copy (even brain vs brain), so setting the
        # limit here only affects this snake (in memory, files are not edited).
        if "TIME_LIMIT_MS" in choose_move.__globals__:
            choose_move.__globals__["TIME_LIMIT_MS"] = time_ms
        _brains.append((label, choose_move))


def play_one(task):
    """Play one game. Returns (winner label or None, {label: move times in ms})."""
    game, base_seed = task
    (label_a, brain_a), (label_b, brain_b) = _brains
    # Swap sides every game; each pair of games shares the same start layout.
    if game % 2 == 0:
        brains, names = [brain_a, brain_b], [label_a, label_b]
    else:
        brains, names = [brain_b, brain_a], [label_b, label_a]
    setup_rng = random.Random(base_seed + game // 2)
    timings = {}
    winner = simulate.play_game(f"sim-{game}", brains, names, setup_rng, random.Random(), timings)
    return (names[winner] if winner is not None else None), timings


def read_option(args, flag, default):
    """Remove '--flag VALUE' from args and return VALUE as an int (or the default)."""
    if flag in args:
        i = args.index(flag)
        value = int(args[i + 1])
        del args[i:i + 2]
        return value
    return default


def main():
    args = sys.argv[1:]
    time_ms = read_option(args, "--time", DEFAULT_TIME_MS)
    time_a = read_option(args, "--time-a", time_ms)
    time_b = read_option(args, "--time-b", time_ms)
    procs = read_option(args, "--procs", max(1, (os.cpu_count() or 1) - 2))
    if len(args) != 3:
        sys.exit("Usage: python simulate_fast.py brain_a brain_b N "
                 "[--time MS] [--time-a MS] [--time-b MS] [--procs P]")
    name_a, name_b, n_games = args[0], args[1], int(args[2])
    if name_a == name_b:  # keep the two sides apart in the results
        label_a, label_b = name_a + " (A)", name_b + " (B)"
    else:
        label_a, label_b = name_a, name_b

    if time_a == time_b:
        print(f"{n_games} games, {time_a} ms per move, {procs} processes")
    else:
        print(f"{n_games} games, {label_a} {time_a} ms / {label_b} {time_b} ms per move, "
              f"{procs} processes")
    wins = {label_a: 0, label_b: 0}
    draws = 0
    times = {label_a: [], label_b: []}
    base_seed = random.randrange(10**9)
    tasks = [(g, base_seed) for g in range(n_games)]
    start = time.perf_counter()

    with multiprocessing.Pool(
        procs, initializer=init_worker,
        initargs=(name_a, name_b, label_a, label_b, time_a, time_b),
    ) as pool:
        for done, (winner, timings) in enumerate(pool.imap_unordered(play_one, tasks), 1):
            if winner is None:
                draws += 1
            else:
                wins[winner] += 1
            for label, t in timings.items():
                times[label].extend(t)
            # Progress counter, rewritten in place on the same line.
            print(f"\r  {done}/{n_games} games done  ({time.perf_counter() - start:.0f} s)",
                  end="", flush=True)

    # Same summary as simulate.py.
    print(f"\n\nGames played: {n_games}")
    for label in (label_a, label_b):
        t = times[label] or [0]
        print(f"{label:>20}: {wins[label]:4d} wins   "
              f"avg {sum(t) / len(t):7.2f} ms/move   max {max(t):7.2f} ms/move")
    print(f"{'draws':>20}: {draws:4d}")

    # Win rate of brain_b with a 95% confidence interval (Wilson score interval,
    # which stays sensible even at 0% or 100%, unlike the plain p +/- 1.96*sqrt(...)).
    n, p, z = n_games, wins[label_b] / n_games, 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    print(f"\n{label_b} win rate: {100 * p:.1f}%  (95% confidence: "
          f"{100 * (center - margin):.1f}% to {100 * (center + margin):.1f}%, "
          f"about +/- {100 * margin:.1f}%)")
    print(f"Total time: {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
