from collections import deque

DIRS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

WIN = 1_000_000
LOSS = -1_000_000
DRAW = -500_000  # both die: better than losing, much worse than surviving
HUNGRY = 40


def step(snake, move):
    """Return a copy of the snake after one move (no eating yet)."""
    dx, dy = DIRS[move]
    hx, hy = snake["body"][0]
    return {"body": [(hx + dx, hy + dy)] + snake["body"][:-1],
            "health": snake["health"] - 1}


def simulate(me, opp, my_move, opp_move, food, w, h):
    """Play one turn by the standard rules. Returns (me, opp, my_dead, opp_dead, food)."""
    me, opp = step(me, my_move), step(opp, opp_move)

    # Eating: full health and grow by one (tail is doubled up).
    food = set(food)
    for s in (me, opp):
        if s["body"][0] in food:
            s["health"] = 100
            s["body"].append(s["body"][-1])
    food -= {me["body"][0], opp["body"][0]}

    def dies(s, other):
        x, y = s["body"][0]
        if s["health"] <= 0 or not (0 <= x < w and 0 <= y < h):
            return True
        if s["body"][0] in s["body"][1:] or s["body"][0] in other["body"][1:]:
            return True
        # Head-to-head: the shorter snake dies, equal length both die.
        return s["body"][0] == other["body"][0] and len(s["body"]) <= len(other["body"])

    return me, opp, dies(me, opp), dies(opp, me), food


def blocked_squares(snakes):
    """Body squares that stay blocked next turn (a tail moves away unless it just grew)."""
    blocked = set()
    for s in snakes:
        body = s["body"]
        if len(body) > 1 and body[-1] != body[-2]:
            body = body[:-1]
        blocked.update(body)
    return blocked


def distances(start, blocked, w, h):
    """BFS distance from start to every reachable free square."""
    dist = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRS.values():
            n = (x + dx, y + dy)
            if 0 <= n[0] < w and 0 <= n[1] < h and n not in blocked and n not in dist:
                dist[n] = dist[(x, y)] + 1
                queue.append(n)
    return dist


def evaluate(me, opp, food, w, h):
    """Score a position where both snakes are alive. Higher is better for me."""
    blocked = blocked_squares([me, opp])
    my_head, opp_head = me["body"][0], opp["body"][0]
    my_dist = distances(my_head, blocked, w, h)
    opp_dist = distances(opp_head, blocked, w, h)
    my_len, opp_len = len(me["body"]), len(opp["body"])

    # Voronoi territory: squares I reach first minus squares they reach first.
    mine = sum(1 for sq, d in my_dist.items() if d < opp_dist.get(sq, 999))
    theirs = sum(1 for sq, d in opp_dist.items() if d < my_dist.get(sq, 999))
    score = (mine - theirs) * 10

    # Getting boxed into a space smaller than my body is close to death.
    if len(my_dist) < my_len:
        score -= 5000
    if len(opp_dist) < opp_len:
        score += 2000

    # Food: only chase it when hungry or shorter than the opponent.
    # Aim for food I reach before the opponent; if there is none, any food I can reach.
    food_dists = ([my_dist[f] for f in food if f in my_dist and my_dist[f] < opp_dist.get(f, 999)]
                  or [my_dist[f] for f in food if f in my_dist])
    nearest = min(food_dists) if food_dists else None
    if me["health"] < HUNGRY:
        weight = 10 + (HUNGRY - me["health"])  # the hungrier, the stronger the pull
    elif my_len < opp_len:
        weight = 6
    else:
        weight = 0
    score -= (nearest if nearest is not None else 25) * weight
    if nearest is None or nearest > me["health"]:
        score -= 3000  # cannot reach food in time

    # Length: reward growing until I am 2 longer (strictly longer wins head-to-heads).
    # Big enough that eating beats the jump in distance to the next food.
    score += max(-5, min(my_len - opp_len, 2)) * 80

    # Head-to-head: if both heads can reach the same square next turn...
    head_gap = abs(my_head[0] - opp_head[0]) + abs(my_head[1] - opp_head[1])
    if opp_len >= my_len and head_gap <= 2:
        score -= 400  # ...stay away from an equal or longer snake
    elif my_len > opp_len:
        score -= head_gap * 8  # ...or hunt a shorter snake down
    return score


def to_snake(s):
    return {"body": [(b["x"], b["y"]) for b in s["body"]], "health": s["health"]}


def choose_move(state):
    board = state["board"]
    w, h = board["width"], board["height"]
    food = {(f["x"], f["y"]) for f in board["food"]}
    me = to_snake(state["you"])
    others = [to_snake(s) for s in board["snakes"] if s["id"] != state["you"]["id"]]
    opp = others[0] if others else {"body": [(-10, -10)], "health": 100}  # dummy if alone

    best_move, best_score = "up", float("-inf")
    for my_move in DIRS:
        # Assume the opponent answers with the reply that is worst for me.
        worst = float("inf")
        for opp_move in DIRS:
            new_me, new_opp, i_die, they_die, new_food = simulate(
                me, opp, my_move, opp_move, food, w, h)
            if i_die and they_die:
                score = DRAW
            elif i_die:
                score = LOSS
            elif they_die and others:
                # A reply that kills them is one they would never pick, unless all do.
                score = WIN
            else:
                score = evaluate(new_me, new_opp, new_food, w, h)
            worst = min(worst, score)
        if worst > best_score:
            best_move, best_score = my_move, worst
    return best_move
