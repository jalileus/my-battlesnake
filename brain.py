from collections import deque

DIRS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}


def flood_fill(start, blocked, w, h):
    """Count how many squares we can reach from start."""
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRS.values():
            n = (x + dx, y + dy)
            if 0 <= n[0] < w and 0 <= n[1] < h and n not in blocked and n not in seen:
                seen.add(n)
                queue.append(n)
    return len(seen)


def choose_move(state):
    board = state["board"]
    w, h = board["width"], board["height"]
    me = state["you"]
    head = (me["head"]["x"], me["head"]["y"])
    my_len = len(me["body"])

    # Squares that are blocked: every snake's body (tails move away, so skip them)
    blocked = set()
    for s in board["snakes"]:
        body = [(b["x"], b["y"]) for b in s["body"]]
        if len(body) > 1 and body[-1] != body[-2]:
            body = body[:-1]
        blocked.update(body)

    # Squares where a bigger or equal snake could hit us head-on
    danger = set()
    for s in board["snakes"]:
        if s["id"] != me["id"] and len(s["body"]) >= my_len:
            hx, hy = s["head"]["x"], s["head"]["y"]
            for dx, dy in DIRS.values():
                danger.add((hx + dx, hy + dy))

    food = [(f["x"], f["y"]) for f in board["food"]]
    hungry = me["health"] < 50

    best_move, best_score = "up", float("-inf")
    for name, (dx, dy) in DIRS.items():
        n = (head[0] + dx, head[1] + dy)
        if not (0 <= n[0] < w and 0 <= n[1] < h) or n in blocked:
            continue  # wall or body: never go here

        space = flood_fill(n, blocked, w, h)
        score = min(space, my_len * 2)  # having enough room is what matters
        if space < my_len:
            score -= 500  # likely a trap
        if n in danger:
            score -= 300  # possible head-on loss
        if food:
            dist = min(abs(n[0] - fx) + abs(n[1] - fy) for fx, fy in food)
            score -= dist * (5 if hungry else 1)

        if score > best_score:
            best_move, best_score = name, score

    return best_move
