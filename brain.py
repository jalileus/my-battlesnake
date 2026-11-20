"""Duel brain v3: iterative-deepening minimax with alpha-beta pruning.

Each ply: I pick a move, the opponent picks the reply that is worst for me,
then both moves are applied together with the standard Battlesnake rules.
"""

import time
from collections import deque

TIME_LIMIT_MS = 100  # thinking budget per move (modified manually to 100 for safety will change if arena does not give strong timeouts)
SAFETY = 0.85  # use at most this part of the budget, the rest is slack
MAX_DEPTH = 40

DIRS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

# Every tunable number of the evaluation, in one place (tune.py changes these in memory).
WEIGHTS = {
    # Game results
    "win": 1_000_000,  # score for a won position
    "loss": -1_000_000,  # score for a lost position
    "draw": -500_000,  # both die: better than losing, much worse than surviving
    # Territory
    "territory": 10,  # per square I reach first (minus squares they reach first)
    "trapped_me": 5000,  # penalty: my reachable space is smaller than my body
    "trapped_opp": 2000,  # bonus: their reachable space is smaller than their body
    # Food
    "hungry": 40,  # below this health, chase food
    "food_hungry_base": 10,  # food pull when hungry (plus 1 per health point below "hungry")
    "food_shorter": 6,  # food pull per square when I am shorter than the opponent
    "no_food_distance": 25,  # distance used when no food is reachable
    "starving": 3000,  # penalty: no food reachable before my health runs out
    # Length
    "length": 160,  # per length point I am ahead (within the limits below)
    "length_max_lead": 2,  # stop rewarding length once I am this much longer
    "length_max_behind": 5,  # stop penalizing length once I am this much shorter
    # Health
    "health_divisor": 10,  # health difference is divided by this (small term)
    # Head-to-head
    "h2h_danger_gap": 2,  # head distance at which an equal/longer snake is dangerous
    "h2h_danger": 0,  # penalty for being that close to an equal/longer snake
    "hunt": 8,  # per square of head distance, when I am longer (hunt them down)
}


class OutOfTime(Exception):
    pass


# ---------------------------------------------------------------- board helpers

_neighbor_cache = {}


def neighbors(w, h):
    """Map each square to {move: neighbor square} (only squares on the board)."""
    if (w, h) not in _neighbor_cache:
        table = {}
        for x in range(w):
            for y in range(h):
                table[(x, y)] = {
                    m: (x + dx, y + dy)
                    for m, (dx, dy) in DIRS.items()
                    if 0 <= x + dx < w and 0 <= y + dy < h
                }
        _neighbor_cache[(w, h)] = table
    return _neighbor_cache[(w, h)]


# A snake is a tuple (body, health); body is a tuple of (x, y), head first.

def play_turn(me, opp, my_move, opp_move, food, nb):
    """Apply both moves with the standard rules.
    Returns (me, opp, my_dead, opp_dead, food)."""
    snakes = []
    for (body, health), move in ((me, my_move), (opp, opp_move)):
        dx, dy = DIRS[move]
        head = (body[0][0] + dx, body[0][1] + dy)
        body = (head,) + body[:-1]  # move: new head, tail moves away
        health -= 1
        if head in food:  # eat: full health and grow (tail is doubled up)
            health = 100
            body = body + (body[-1],)
        snakes.append((body, health))
    me, opp = snakes
    eaten = {me[0][0], opp[0][0]} & food
    if eaten:
        food = food - eaten

    def dies(s, other):
        body, health = s
        head = body[0]
        if health <= 0 or head not in nb:  # starved or hit a wall
            return True
        if head in body[1:] or head in other[0][1:]:  # hit a body
            return True
        # Head-to-head: the shorter snake dies, equal length both die.
        return head == other[0][0] and len(body) <= len(other[0])

    return me, opp, dies(me, opp), dies(opp, me), food


def blocked_squares(me, opp):
    """Body squares that stay blocked next turn (a tail moves away unless it just grew)."""
    blocked = set()
    for body, _ in (me, opp):
        if len(body) > 1 and body[-1] != body[-2]:
            body = body[:-1]
        blocked.update(body)
    return blocked


def distances(start, blocked, nb):
    """BFS distance from start to every reachable free square."""
    dist = {start: 0}
    queue = deque([start])
    while queue:
        sq = queue.popleft()
        d = dist[sq] + 1
        for n in nb[sq].values():
            if n not in blocked and n not in dist:
                dist[n] = d
                queue.append(n)
    return dist


# ---------------------------------------------------------------- evaluation

def evaluate(me, opp, food, nb):
    """Score a position where both snakes are alive. Higher is better for me."""
    blocked = blocked_squares(me, opp)
    my_head, opp_head = me[0][0], opp[0][0]
    my_dist = distances(my_head, blocked, nb)
    opp_dist = distances(opp_head, blocked, nb)
    my_len, opp_len = len(me[0]), len(opp[0])
    my_health = me[1]
    W = WEIGHTS

    # Voronoi territory: squares I reach first minus squares they reach first.
    mine = sum(1 for sq, d in my_dist.items() if d < opp_dist.get(sq, 999))
    theirs = sum(1 for sq, d in opp_dist.items() if d < my_dist.get(sq, 999))
    score = (mine - theirs) * W["territory"]

    # Getting boxed into a space smaller than my body is close to death.
    if len(my_dist) < my_len:
        score -= W["trapped_me"]
    if len(opp_dist) < opp_len:
        score += W["trapped_opp"]

    # Food: only chase it when hungry or shorter than the opponent.
    # Aim for food I reach first; if there is none, any food I can reach.
    food_dists = [
        my_dist[f] for f in food if f in my_dist and my_dist[f] < opp_dist.get(f, 999)
    ] or [my_dist[f] for f in food if f in my_dist]
    nearest = min(food_dists) if food_dists else None
    if my_health < W["hungry"]:
        # the hungrier, the stronger the pull
        weight = W["food_hungry_base"] + (W["hungry"] - my_health)
    elif my_len < opp_len:
        weight = W["food_shorter"]
    else:
        weight = 0
    score -= (nearest if nearest is not None else W["no_food_distance"]) * weight
    if nearest is None or nearest > my_health:
        score -= W["starving"]  # cannot reach food in time

    # Length: reward growing until I am a bit longer (strictly longer wins head-to-heads).
    lead = max(-W["length_max_behind"], min(my_len - opp_len, W["length_max_lead"]))
    score += lead * W["length"]

    # Small health term: a bit of spare health is nice, but not worth much.
    score += (my_health - opp[1]) // W["health_divisor"]

    # Head-to-head: if both heads can reach the same square next turn...
    head_gap = abs(my_head[0] - opp_head[0]) + abs(my_head[1] - opp_head[1])
    if opp_len >= my_len and head_gap <= W["h2h_danger_gap"]:
        score -= W["h2h_danger"]  # ...stay away from an equal or longer snake
    elif my_len > opp_len:
        score -= head_gap * W["hunt"]  # ...or hunt a shorter snake down
    return score


# ---------------------------------------------------------------- search

def search_move(state):
    start = time.perf_counter()
    deadline = start + TIME_LIMIT_MS * SAFETY / 1000

    board = state["board"]
    nb = neighbors(board["width"], board["height"])
    food = frozenset((f["x"], f["y"]) for f in board["food"])
    you_id = state["you"]["id"]

    def to_snake(s):
        return tuple((b["x"], b["y"]) for b in s["body"]), s["health"]

    me = to_snake(state["you"])
    opp = to_snake(next(s for s in board["snakes"] if s["id"] != you_id))

    # Best moves found by the previous depth, per position in the tree (for move ordering).
    my_best = {}  # path -> my best move
    opp_best = {}  # (path, my move) -> opponent's strongest reply

    def ordered(first):
        moves = list(DIRS)
        if first in moves:
            moves.remove(first)
            moves.insert(0, first)
        return moves

    def max_node(me, opp, food, depth, alpha, beta, path):
        """Value of this position for me, searching `depth` more plies."""
        best, best_move = float("-inf"), None
        for my_move in ordered(my_best.get(path)):
            # Min part: the opponent answers with the reply that is worst for me.
            worst, worst_reply = float("inf"), None
            for opp_move in ordered(opp_best.get((path, my_move))):
                if time.perf_counter() > deadline:
                    raise OutOfTime  # stop right away, keep the last completed depth
                new_me, new_opp, i_die, they_die, new_food = play_turn(
                    me, opp, my_move, opp_move, food, nb
                )
                # Earlier wins and later losses are better (depth = plies left).
                if i_die and they_die:
                    score = WEIGHTS["draw"]
                elif i_die:
                    score = WEIGHTS["loss"] - depth
                elif they_die:
                    score = WEIGHTS["win"] + depth
                elif depth == 1:
                    score = evaluate(new_me, new_opp, new_food, nb)
                else:
                    score = max_node(
                        new_me, new_opp, new_food, depth - 1,
                        alpha, min(beta, worst), path + ((my_move, opp_move),),
                    )
                if score < worst:
                    worst, worst_reply = score, opp_move
                if worst <= alpha:
                    break  # this move can't beat one I already have
            opp_best[(path, my_move)] = worst_reply

            if worst > best:
                best, best_move = worst, my_move
            alpha = max(alpha, best)
            if alpha >= beta:
                break  # the opponent would never let us reach this position
        my_best[path] = best_move
        return best

    best_move = None
    for depth in range(1, MAX_DEPTH + 1):
        depth_start = time.perf_counter()
        try:
            score = max_node(me, opp, food, depth, float("-inf"), float("inf"), ())
        except OutOfTime:
            break  # keep the move from the last completed depth
        best_move = my_best[()]

        if abs(score) >= WEIGHTS["win"] - 1000:
            break  # forced win or loss found: searching deeper changes nothing
        # The next depth takes several times longer; don't start it if it can't finish.
        now = time.perf_counter()
        if now + (now - depth_start) * 3 > deadline:
            break
    return best_move


# ---------------------------------------------------------------- fallback + entry point

def fallback_move(state):
    """The older brain_v2 logic, or any move that doesn't hit a wall or body."""
    try:
        import brain_v2

        return brain_v2.choose_move(state)
    except Exception:
        board = state["board"]
        head = state["you"]["head"]
        taken = {(b["x"], b["y"]) for s in board["snakes"] for b in s["body"][:-1]}
        for move, (dx, dy) in DIRS.items():
            x, y = head["x"] + dx, head["y"] + dy
            if 0 <= x < board["width"] and 0 <= y < board["height"] and (x, y) not in taken:
                return move
        return "up"


def choose_move(state):
    try:
        if len(state["board"]["snakes"]) != 2:
            return fallback_move(state)
        move = search_move(state)
        return move if move else fallback_move(state)
    except Exception:
        return fallback_move(state)
