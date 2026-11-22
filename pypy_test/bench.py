"""Speed benchmark for brain.py, run the same way under CPython and PyPy.

Order matters (PyPy gets faster while it runs, CPython does not):
  1. cold:   first 50 of the 400-position sample at 100 ms (fresh process)
  2. early:  the 400 positions at 20 ms and at 100 ms
  3. fixed:  all 11,300 positions at fixed depth 4 (moves are compared later)
  4. warm:   the 400 positions at 20 ms and at 100 ms again
  5. warm50: the same first 50 positions at 100 ms again
The last line of output is RESULT_JSON {...} for compare.py.
"""

import json
import platform
import sys
import time
import zlib

import brain

positions = json.load(open("data/positions.json", encoding="utf-8"))
sample = positions[::28][:400]  # same 400 positions as the earlier depth tests
MOVES = "udlr"  # move number -> letter (up, down, left, right)
clock = time.perf_counter


def timed_run(states, limit_ms):
    """Search each position at a real time limit. Returns summary numbers."""
    brain.TIME_LIMIT_MS, brain.MAX_DEPTH = limit_ms, 40
    depths, total_pos, times = [], 0, []
    for s in states:
        t = clock()
        _, depth, pos = brain.search_move(s)
        times.append((clock() - t) * 1000)
        depths.append(depth)
        total_pos += pos
    return {
        "moves": len(states),
        "avg_depth": sum(depths) / len(depths),
        "pos_per_s": total_pos / (sum(times) / 1000),
        "avg_ms": sum(times) / len(times),
        "max_ms": max(times),
        "over_limit": sum(1 for x in times if x > limit_ms),
    }


def fixed_depth_run(states, depth):
    """Search each position to exactly `depth` with no time limit."""
    brain.TIME_LIMIT_MS, brain.MAX_DEPTH = 10**9, depth
    moves, counts = [], []
    t = clock()
    for s in states:
        move, _, pos = brain.search_move(s)
        moves.append(MOVES[move] if move is not None else "-")
        counts.append(pos)
    seconds = clock() - t
    return {
        "positions": len(states),
        "seconds": seconds,
        "pos_per_s": sum(counts) / seconds,
        "ms_per_position": 1000 * seconds / len(states),
        "moves": "".join(moves),
        "counts_crc": zlib.crc32(",".join(map(str, counts)).encode()),
        "total_searched": sum(counts),
    }


def show(name, r):
    print(f"{name:<14} depth {r['avg_depth']:.2f}  {r['pos_per_s']:>9,.0f} pos/s  "
          f"avg {r['avg_ms']:6.1f} ms  max {r['max_ms']:6.1f} ms  over limit {r['over_limit']}",
          flush=True)


result = {"python": f"{platform.python_implementation()} {platform.python_version()}",
          "build": sys.version.replace("\n", " ")}
print(result["python"], "|", result["build"], flush=True)

result["cold50_100"] = timed_run(sample[:50], 100)
show("cold 50 @100", result["cold50_100"])
result["early_20"] = timed_run(sample, 20)
show("early 400 @20", result["early_20"])
result["early_100"] = timed_run(sample, 100)
show("early 400 @100", result["early_100"])

result["fixed4"] = fixed_depth_run(positions, 4)
f = result["fixed4"]
print(f"fixed depth 4  {f['positions']} positions in {f['seconds']:.1f} s  "
      f"{f['pos_per_s']:,.0f} pos/s  {f['ms_per_position']:.2f} ms each", flush=True)

result["warm_20"] = timed_run(sample, 20)
show("warm 400 @20", result["warm_20"])
result["warm_100"] = timed_run(sample, 100)
show("warm 400 @100", result["warm_100"])
result["warm50_100"] = timed_run(sample[:50], 100)
show("warm 50 @100", result["warm50_100"])

print("RESULT_JSON " + json.dumps(result))
