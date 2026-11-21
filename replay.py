"""Replay saved /move requests through brain.py to see why a game was lost.

Usage:
    python replay.py [folder]        (default folder: lost_games_turns)

Each .json file is one /move request (keys game, turn, board, you). A file may
also hold a list of such requests, or wrap one under a key like "request".
For every turn it prints the board, the move at the real time limit, every
move's search score, a much deeper search for comparison, and the space left
after each move. A summary table comes at the end.
"""

import json
import os
import sys
from collections import deque

import brain

REAL_TIME_MS = 100
DEEP_TIME_MS = 2000
DIRS = brain.DIRS


# ---------------------------------------------------------------- loading

def find_requests(data):
    """Return the /move requests inside a loaded JSON value (dict, list or wrapper)."""
    if isinstance(data, list):
        return [r for item in data for r in find_requests(item)]
    if isinstance(data, dict):
        if "board" in data and "you" in data:
            return [data]
        for value in data.values():  # wrapper: look one level deeper
            if isinstance(value, (dict, list)):
                found = find_requests(value)
                if found:
                    return found
    return []


def load_turns(folder):
    """List of (label, request) for every .json file, sorted by file name."""
    turns = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(folder, name), encoding="utf-8") as f:
            requests = find_requests(json.load(f))
        if not requests:
            print(f"!! {name}: no /move request found, skipped")
        for i, req in enumerate(requests):
            label = name if len(requests) == 1 else f"{name}[{i}]"
            turns.append((label, req))
    return turns


# ---------------------------------------------------------------- board helpers

def xy(p):
    return p["x"], p["y"]


def draw_board(state):
    """Text picture, top row first: M/m me, O/o opponent, F food, . empty."""
    board = state["board"]
    w, h = board["width"], board["height"]
    grid = [["." for _ in range(w)] for _ in range(h)]
    for f in board["food"]:
        grid[f["y"]][f["x"]] = "F"
    you_id = state["you"]["id"]
    for s in board["snakes"]:
        mine = s["id"] == you_id
        for p in s["body"][1:]:
            grid[p["y"]][p["x"]] = "m" if mine else "o"
        grid[s["head"]["y"]][s["head"]["x"]] = "M" if mine else "O"
    lines = []
    for y in range(h - 1, -1, -1):
        lines.append(f"  {y:2d} " + " ".join(grid[y]))
    lines.append("     " + " ".join(str(x % 10) for x in range(w)))
    return "\n".join(lines)


def blocked_after_move(state):
    """Body squares still blocked next turn (tails move away unless the snake just ate)."""
    blocked = set()
    for s in state["board"]["snakes"]:
        body = [xy(p) for p in s["body"]]
        if len(body) > 1 and body[-1] != body[-2]:
            body = body[:-1]
        blocked.update(body)
    return blocked


def move_info(state):
    """For each move: (legal?, reason, reachable area after the move)."""
    board = state["board"]
    w, h = board["width"], board["height"]
    hx, hy = xy(state["you"]["head"])
    blocked = blocked_after_move(state)
    info = {}
    for move, (dx, dy) in DIRS.items():
        nx, ny = hx + dx, hy + dy
        if not (0 <= nx < w and 0 <= ny < h):
            info[move] = (False, "wall", 0)
        elif (nx, ny) in blocked:
            info[move] = (False, "body", 0)
        else:
            # My head is now on (nx, ny) and my old head becomes body (still blocked).
            info[move] = (True, "", flood_fill((nx, ny), blocked, w, h))
    return info


def flood_fill(start, blocked, w, h):
    """Number of free squares reachable from start (start itself not counted)."""
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRS.values():
            n = (x + dx, y + dy)
            if 0 <= n[0] < w and 0 <= n[1] < h and n not in blocked and n not in seen:
                seen.add(n)
                queue.append(n)
    return len(seen) - 1


def describe_score(score, depth):
    """Readable label for win/loss scores (they encode how many plies ahead)."""
    win, loss, draw = brain.WEIGHTS["win"], brain.WEIGHTS["loss"], brain.WEIGHTS["draw"]
    if score == draw:
        return "DRAW (both die)"
    if score <= loss + 1000:
        return f"LOSS in {depth - (loss - score) + 1} turn(s)"
    if score >= win - 1000:
        return f"WIN in {depth - (score - win) + 1} turn(s)"
    return ""


# ---------------------------------------------------------------- search

def search(state, time_ms, with_scores):
    """Run brain's own search with a temporary time limit (in memory, then restored)."""
    old = brain.TIME_LIMIT_MS
    brain.TIME_LIMIT_MS = time_ms
    try:
        return brain.debug_search(state, with_scores)
    finally:
        brain.TIME_LIMIT_MS = old


# ---------------------------------------------------------------- main

def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "lost_games_turns"
    turns = load_turns(folder)
    summary = []

    for label, state in turns:
        you = state["you"]
        opps = [s for s in state["board"]["snakes"] if s["id"] != you["id"]]
        opp = opps[0] if opps else None
        print("=" * 78)
        print(f"{label}")
        print(f"turn {state['turn']}   my length {you['length']}   "
              f"opponent length {opp['length'] if opp else '-'}   my health {you['health']}"
              + (f"   (opponent health {opp['health']})" if opp else ""))
        print(draw_board(state))

        if len(state["board"]["snakes"]) != 2:
            # brain.py only searches 1v1; other boards go to its fallback.
            move = brain.choose_move(state)
            print(f"\n{len(state['board']['snakes'])} snakes on the board: "
                  f"brain uses its fallback, move {move}")
            summary.append((label, state["turn"], move, "-", "-", 0))
            continue

        real = search(state, REAL_TIME_MS, with_scores=True)
        deep = search(state, DEEP_TIME_MS, with_scores=False)
        info = move_info(state)

        print(f"\nmove at {REAL_TIME_MS} ms: {real['move']}   "
              f"(depth {real['depth']}, {real['positions']} positions)")
        print(f"  {'move':<6} {'legal':<11} {'score':>10}  {'area after move':>15}")
        for move in DIRS:
            legal, reason, area = info[move]
            score = real["scores"].get(move) if real["scores"] else None
            note = describe_score(score, real["depth"]) if score is not None else ""
            pick = "  <- chosen" if move == real["move"] else ""
            print(f"  {move:<6} {'yes' if legal else 'no (' + reason + ')':<11} "
                  f"{score if score is not None else '-':>10}  "
                  f"{area if legal else '-':>15}  {note}{pick}")

        different = deep["move"] != real["move"]
        print(f"\ndeep search ({DEEP_TIME_MS} ms): {deep['move']}   (depth {deep['depth']})"
              + ("   <<< DIFFERENT" if different else "   (same move)"))
        summary.append((label, state["turn"], real["move"], deep["move"],
                        "yes" if different else "no", real["depth"]))

    # Summary table, one row per file.
    print("\n" + "=" * 78)
    print("SUMMARY")
    width = max([len(row[0]) for row in summary] + [4])
    print(f"{'file':<{width}}  {'turn':>4}  {'100ms':<6} {'2000ms':<6} {'DIFFERENT':<9} depth@100ms")
    for label, turn, real_move, deep_move, diff, depth in summary:
        print(f"{label:<{width}}  {turn:>4}  {str(real_move):<6} {str(deep_move):<6} "
              f"{diff:<9} {depth}")
    print(f"\n{len(summary)} turns, {sum(1 for row in summary if row[4] == 'yes')} "
          f"where the deep search picks a different move")


if __name__ == "__main__":
    main()
