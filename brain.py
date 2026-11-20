"""Duel brain: iterative-deepening minimax with alpha-beta pruning (speed-optimized v5).

Each ply: I pick a move, the opponent picks the reply that is worst for me,
then both moves are applied together with the standard Battlesnake rules.

Speed tricks (the moves chosen are exactly the same as brain_v5):
- squares are numbers (i = y * width + x) and snakes are tuples of numbers
- moves use a precomputed table: step[square][move] -> next square (-1 = wall)
- food and walls/bodies are bitmasks (one bit per square in a Python int), so the
  territory BFS grows a whole layer at once with a few shifts instead of one
  square at a time
- move ordering uses small precomputed tuples, and tree paths are single numbers
"""

import time

TIME_LIMIT_MS = 100  # thinking budget per move (modified manually to 100 for safety will change if arena does not give strong timeouts)
SAFETY = 0.85  # use at most this part of the budget, the rest is slack
MAX_DEPTH = 40

DIRS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
MOVE_NAMES = tuple(DIRS)  # move number -> name: 0 up, 1 down, 2 left, 3 right

# Every tunable number of the evaluation, in one place (tune.py changes these in memory).
WEIGHTS = {
    # Game results
    "win": 1_000_000,  # score for a won position
    "loss": -1_000_000,  # score for a lost position
    "draw": -500_000,  # both die: better than losing, much worse than surviving
    # Territory
    "territory": 6,  # per square I reach first (minus squares they reach first)
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
    "length_max_lead": 4,  # stop rewarding length once I am this much longer
    "length_max_behind": 5,  # stop penalizing length once I am this much shorter
    # Health
    "health_divisor": 10,  # health difference is divided by this (small term)
    # Head-to-head
    "h2h_danger_gap": 2,  # head distance at which an equal/longer snake is dangerous
    "h2h_danger": 0,  # penalty for being that close to an equal/longer snake
    "hunt": 8,  # per square of head distance, when I am longer (hunt them down)
}

# Move order: try `first` first, then the rest in the normal order.
ORDER = ((0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3), (3, 0, 1, 2))


class OutOfTime(Exception):
    pass


# ---------------------------------------------------------------- board tables

class Board:
    """Lookup tables for a w x h board, built once per board size."""

    def __init__(self, w, h):
        self.w, n = w, w * h
        self.x = tuple(i % w for i in range(n))
        self.y = tuple(i // w for i in range(n))
        self.bit = tuple(1 << i for i in range(n))
        # step[i][m]: the square reached from i with move m, or -1 for a wall.
        self.step = tuple(
            tuple(
                (i // w + dy) * w + (i % w + dx)
                if 0 <= i % w + dx < w and 0 <= i // w + dy < h else -1
                for dx, dy in DIRS.values()
            )
            for i in range(n)
        )
        self.full = (1 << n) - 1  # every square on the board
        # Masks that stop a shift left/right from wrapping into the next row.
        self.not_col0 = self.full & ~sum(1 << (y * w) for y in range(h))
        self.not_last_col = self.full & ~sum(1 << (y * w + w - 1) for y in range(h))


_boards = {}


def board_tables(w, h):
    if (w, h) not in _boards:
        _boards[(w, h)] = Board(w, h)
    return _boards[(w, h)]


# ---------------------------------------------------------------- rules

# A snake is a tuple (body, health); body is a tuple of square numbers, head first.
# Food is a bitmask of squares.

def play_turn(me, opp, my_move, opp_move, food, step):
    """Apply both moves with the standard rules.
    Returns (me, opp, my_dead, opp_dead, food)."""
    my_body, my_health = me
    head = step[my_body[0]][my_move]
    my_body = (head,) + my_body[:-1]  # move: new head, tail moves away
    my_health -= 1
    if head >= 0 and food >> head & 1:  # eat: full health and grow (tail doubled up)
        my_health = 100
        my_body += (my_body[-1],)

    opp_body, opp_health = opp
    head = step[opp_body[0]][opp_move]
    opp_body = (head,) + opp_body[:-1]
    opp_health -= 1
    if head >= 0 and food >> head & 1:
        opp_health = 100
        opp_body += (opp_body[-1],)

    for head in (my_body[0], opp_body[0]):  # eaten food disappears
        if head >= 0:
            food &= ~(1 << head)

    def dies(body, health, other):
        head = body[0]
        if health <= 0 or head < 0:  # starved or hit a wall
            return True
        if head in body[1:] or head in other[1:]:  # hit a body
            return True
        # Head-to-head: the shorter snake dies, equal length both die.
        return head == other[0] and len(body) <= len(other)

    return (
        (my_body, my_health), (opp_body, opp_health),
        dies(my_body, my_health, opp_body), dies(opp_body, opp_health, my_body), food,
    )


# ---------------------------------------------------------------- evaluation

def evaluate(me, opp, food, bd):
    """Score a position where both snakes are alive. Higher is better for me."""
    my_body, my_health = me
    opp_body, opp_health = opp
    W = WEIGHTS
    bit = bd.bit

    # Blocked squares: both bodies (a tail moves away next turn unless it just grew).
    blocked = 0
    for body in (my_body, opp_body):
        if len(body) > 1 and body[-1] != body[-2]:
            body = body[:-1]
        for sq in body:
            blocked |= bit[sq]
    free = bd.full & ~blocked

    # Two BFS's at once, one layer (= one distance) per loop, using bitmasks.
    # A square is mine if I reach it at a smaller distance than the opponent.
    my_head, opp_head = my_body[0], opp_body[0]
    my_front = my_seen = bit[my_head]
    opp_front = opp_seen = bit[opp_head]
    my_left = opp_left = free  # free squares each side has not reached yet
    w, not_col0, not_last_col = bd.w, bd.not_col0, bd.not_last_col
    mine = theirs = 0
    food_mine = food_any = None  # distance to nearest food I reach first / any food
    dist = 0
    while my_front or opp_front:
        mine += (my_front & ~opp_seen).bit_count()
        theirs += (opp_front & ~my_seen).bit_count()
        if my_front & food:
            if food_any is None:
                food_any = dist
            if food_mine is None and my_front & food & ~opp_seen:
                food_mine = dist
        # Grow both areas by one step (shifts = left/right/up/down).
        f = my_front
        my_front = (((f << 1) & not_col0) | ((f >> 1) & not_last_col)
                    | (f << w) | (f >> w)) & my_left
        my_left ^= my_front
        my_seen |= my_front
        f = opp_front
        opp_front = (((f << 1) & not_col0) | ((f >> 1) & not_last_col)
                     | (f << w) | (f >> w)) & opp_left
        opp_left ^= opp_front
        opp_seen |= opp_front
        dist += 1

    score = (mine - theirs) * W["territory"]
    my_len, opp_len = len(my_body), len(opp_body)

    # Getting boxed into a space smaller than my body is close to death.
    if my_seen.bit_count() < my_len:
        score -= W["trapped_me"]
    if opp_seen.bit_count() < opp_len:
        score += W["trapped_opp"]

    # Food: only chase it when hungry or shorter than the opponent.
    # Aim for food I reach first; if there is none, any food I can reach.
    nearest = food_mine if food_mine is not None else food_any
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
    score += (my_health - opp_health) // W["health_divisor"]

    # Head-to-head: if both heads can reach the same square next turn...
    head_gap = abs(bd.x[my_head] - bd.x[opp_head]) + abs(bd.y[my_head] - bd.y[opp_head])
    if opp_len >= my_len and head_gap <= W["h2h_danger_gap"]:
        score -= W["h2h_danger"]  # ...stay away from an equal or longer snake
    elif my_len > opp_len:
        score -= head_gap * W["hunt"]  # ...or hunt a shorter snake down
    return score


# ---------------------------------------------------------------- search

def search_move(state):
    """Returns (best move number or None, last completed depth, positions searched)."""
    start = time.perf_counter()
    deadline = start + TIME_LIMIT_MS * SAFETY / 1000
    clock = time.perf_counter

    board = state["board"]
    w = board["width"]
    bd = board_tables(w, board["height"])
    step = bd.step
    food = 0
    for f in board["food"]:
        food |= 1 << (f["y"] * w + f["x"])
    you_id = state["you"]["id"]

    def to_snake(s):
        return tuple(b["y"] * w + b["x"] for b in s["body"]), s["health"]

    me = to_snake(state["you"])
    opp = to_snake(next(s for s in board["snakes"] if s["id"] != you_id))

    win, loss, draw = WEIGHTS["win"], WEIGHTS["loss"], WEIGHTS["draw"]
    # Best moves found by the previous depth, per position in the tree (for move ordering).
    # A path (the moves from the root) is stored as one number: path * 17 + (1..16).
    my_best = {}  # path -> my best move
    opp_best = {}  # path * 4 + my move -> opponent's strongest reply
    positions = 0

    def max_node(me, opp, food, depth, alpha, beta, path):
        """Value of this position for me, searching `depth` more plies."""
        nonlocal positions
        best, best_move = float("-inf"), None
        for my_move in ORDER[my_best.get(path, 0)]:
            # Min part: the opponent answers with the reply that is worst for me.
            worst, worst_reply = float("inf"), None
            key = path * 4 + my_move
            for opp_move in ORDER[opp_best.get(key, 0)]:
                if clock() > deadline:
                    raise OutOfTime  # stop right away, keep the last completed depth
                positions += 1
                new_me, new_opp, i_die, they_die, new_food = play_turn(
                    me, opp, my_move, opp_move, food, step
                )
                # Earlier wins and later losses are better (depth = plies left).
                if i_die and they_die:
                    score = draw
                elif i_die:
                    score = loss - depth
                elif they_die:
                    score = win + depth
                elif depth == 1:
                    score = evaluate(new_me, new_opp, new_food, bd)
                else:
                    score = max_node(
                        new_me, new_opp, new_food, depth - 1,
                        alpha, min(beta, worst), path * 17 + my_move * 4 + opp_move + 1,
                    )
                if score < worst:
                    worst, worst_reply = score, opp_move
                if worst <= alpha:
                    break  # this move can't beat one I already have
            opp_best[key] = worst_reply

            if worst > best:
                best, best_move = worst, my_move
            alpha = max(alpha, best)
            if alpha >= beta:
                break  # the opponent would never let us reach this position
        my_best[path] = best_move
        return best

    best_move, completed = None, 0
    for depth in range(1, MAX_DEPTH + 1):
        depth_start = clock()
        try:
            score = max_node(me, opp, food, depth, float("-inf"), float("inf"), 0)
        except OutOfTime:
            break  # keep the move from the last completed depth
        best_move, completed = my_best[0], depth

        if abs(score) >= win - 1000:
            break  # forced win or loss found: searching deeper changes nothing
        # The next depth takes several times longer; don't start it if it can't finish.
        now = clock()
        if now + (now - depth_start) * 3 > deadline:
            break
    return best_move, completed, positions


# ---------------------------------------------------------------- debug counters

STATS = {}  # game id -> [moves, total depth, total positions searched]
MAX_GAMES_KEPT = 200


def record_stats(game_id, depth, positions):
    if game_id not in STATS and len(STATS) >= MAX_GAMES_KEPT:
        del STATS[next(iter(STATS))]  # forget the oldest game
    s = STATS.setdefault(game_id, [0, 0, 0])
    s[0] += 1
    s[1] += depth
    s[2] += positions


def print_game_stats(game_id=None, reset=True):
    """Print average search depth and positions per move for one game (or all games).
    For example, call brain.print_game_stats(game_state["game"]["id"]) in end()."""
    ids = [game_id] if game_id is not None else list(STATS)
    moves = sum(STATS[g][0] for g in ids if g in STATS)
    if not moves:
        print("brain stats: no searched moves recorded")
        return None
    depth = sum(STATS[g][1] for g in ids if g in STATS) / moves
    positions = sum(STATS[g][2] for g in ids if g in STATS) / moves
    print(f"brain stats: {moves} moves, average depth {depth:.2f}, "
          f"average {positions:.0f} positions searched per move")
    if reset:
        for g in ids:
            STATS.pop(g, None)
    return {"moves": moves, "avg_depth": depth, "avg_positions": positions}


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
        move, depth, positions = search_move(state)
        record_stats(state.get("game", {}).get("id"), depth, positions)
        return MOVE_NAMES[move] if move is not None else fallback_move(state)
    except Exception:
        return fallback_move(state)
