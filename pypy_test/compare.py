"""Compare the CPython and PyPy benchmark results in results/*.txt and print a table."""

import json


def load(name):
    with open(f"results/{name}.txt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("RESULT_JSON "):
                return json.loads(line[len("RESULT_JSON "):])
    raise SystemExit(f"results/{name}.txt has no RESULT_JSON line (the run failed?)")


c, p = load("cpython"), load("pypy")
print(f"A: {c['python']}    B: {p['python']}\n")

# (a) identical moves at fixed depth 4
mc, mp = c["fixed4"]["moves"], p["fixed4"]["moves"]
diff = [i for i, (x, y) in enumerate(zip(mc, mp)) if x != y]
print(f"fixed depth 4: {len(mc)} vs {len(mp)} positions, "
      f"{len(mc) - len(diff)} same moves, {len(diff)} different"
      + (f" (first: {diff[:10]})" if diff else ""))
print("positions searched per position identical:",
      c["fixed4"]["counts_crc"] == p["fixed4"]["counts_crc"],
      f"({c['fixed4']['total_searched']:,} vs {p['fixed4']['total_searched']:,} in total)\n")

rows = [
    ("fixed depth 4, all 11,300", "fixed4", None),
    ("cold: first 50 @100 ms", "cold50_100", "avg_depth"),
    ("early: 400 @20 ms", "early_20", "avg_depth"),
    ("early: 400 @100 ms", "early_100", "avg_depth"),
    ("warm: 400 @20 ms", "warm_20", "avg_depth"),
    ("warm: 400 @100 ms", "warm_100", "avg_depth"),
    ("warm: same first 50 @100 ms", "warm50_100", "avg_depth"),
]
print(f"{'run':<29} {'CPython pos/s':>13} {'PyPy pos/s':>11} {'k':>6}   "
      f"{'depth C':>7} {'depth P':>7}   {'max ms C':>8} {'max ms P':>8}")
for label, key, _ in rows:
    a, b = c[key], p[key]
    k = b["pos_per_s"] / a["pos_per_s"]
    if key == "fixed4":
        extra = f"{'4':>7} {'4':>7}   {'-':>8} {'-':>8}"
    else:
        extra = (f"{a['avg_depth']:>7.2f} {b['avg_depth']:>7.2f}   "
                 f"{a['max_ms']:>8.1f} {b['max_ms']:>8.1f}")
    print(f"{label:<29} {a['pos_per_s']:>13,.0f} {b['pos_per_s']:>11,.0f} {k:>5.2f}x   {extra}")

print("\nmoves over the time limit (C / P):",
      ", ".join(f"{key} {c[key]['over_limit']}/{p[key]['over_limit']}"
                for _, key, d in rows if d))
