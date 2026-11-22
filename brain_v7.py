"""Duel brain: iterative-deepening minimax with alpha-beta pruning (speed-optimized v7).

Each ply: I pick a move, the opponent picks the reply that is worst for me,
then both moves are applied  together with the standard Battlesnake rules.

Speed tricks (the moves chosen are exactly the same as brain_v6):
- squares are numbers and a snake is (body tuple, health, body bitmask)
- the board is stored one column wider than it is (11x11 -> 12 wide); the extra
  column is never free, so the territory BFS can shift bitmasks without edge masks
- moves use a precomputed table: step[square][move] -> next square (-1 = wall)
- the rules are applied inside the search loop, and each snake's move is worked
  out once per position instead of once per pair of moves
"""

import time

TIME_LIMIT_MS = (
    130  # thinking budget per move (was 100; raised after the v7 speed-up was verified)
)
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
    """Lookup tables for a w x h board, built once per board size.
    Square of (x, y) = y * stride + x, with stride = w + 1: the extra column on the
    right is never free, so a bit shifted off one edge lands on a blocked square."""

    def __init__(self, w, h):
        self.w = w
        self.stride = s = w + 1
        n = s * h
        self.x = tuple(i % s for i in range(n))
        self.y = tuple(i // s for i in range(n))
        self.bit = tuple(1 << i for i in range(n))
        # step[i][m]: the square reached from i with move m, or -1 for a wall.
        self.step = tuple(
            tuple(
                (
                    (i // s + dy) * s + (i % s + dx)
                    if 0 <= i % s + dx < w and 0 <= i // s + dy < h
                    else -1
                )
                for dx, dy in DIRS.values()
            )
            for i in range(n)
        )
        self.full = sum(
            1 << (y * s + x) for y in range(h) for x in range(w)
        )  # real squares

    def square(self, p):
        return p["y"] * self.stride + p["x"]


_boards = {}


def board_tables(w, h):
    if (w, h) not in _boards:
        _boards[(w, h)] = Board(w, h)
    return _boards[(w, h)]


# ---------------------------------------------------------------- rules

# A snake is a tuple (body, health, mask): body is a tuple of squares (head first),
# mask has one bit per body square. Food is a bitmask of squares.


def blocking(body, mask, bit):
    """Squares this snake still blocks next turn: its body minus the tail
    (the tail moves away, unless the snake just ate and it is doubled up)."""
    if len(body) > 1 and body[-1] != body[-2]:
        return mask ^ bit[body[-1]]
    return mask


def move_snake(snake, move, food, blocked, bd):
    """One snake's own move: (new head, snake after the move or None, crashed?).
    `blocked` = squares both bodies block. Head-to-head is checked by the caller."""
    body, health, mask = snake
    head = bd.step[body[0]][move]
    if head < 0:
        return head, None, True  # hit a wall
    ate = food >> head & 1
    health = 100 if ate else health - 1
    if health <= 0 or blocked >> head & 1:
        return head, None, True  # starved or hit a body
    new_body = (head,) + body[:-1]
    if ate:
        new_body += (new_body[-1],)  # grow: the tail doubles up
    return head, (new_body, health, blocking(body, mask, bd.bit) | bd.bit[head]), False


def play_turn(me, opp, my_move, opp_move, food, bd):
    """Apply both moves with the standard rules. The search does the same work
    inline; this is for the debug scores. Returns (me, opp, my_dead, opp_dead, food)."""
    bit = bd.bit
    blocked = blocking(me[0], me[2], bit) | blocking(opp[0], opp[2], bit)
    my_head, new_me, i_die = move_snake(me, my_move, food, blocked, bd)
    opp_head, new_opp, they_die = move_snake(opp, opp_move, food, blocked, bd)
    if my_head == opp_head:
        # Head-to-head: the shorter snake dies, equal length both die.
        # (Both heads are on the same square, so both ate or neither did.)
        i_die = i_die or len(me[0]) <= len(opp[0])
        they_die = they_die or len(opp[0]) <= len(me[0])
    if not (i_die or they_die):
        food &= ~(bit[my_head] | bit[opp_head])  # eaten food disappears
    return new_me, new_opp, i_die, they_die, food


# ---------------------------------------------------------------- evaluation


def evaluate(me, opp, food, bd):
    """Score a position where both snakes are alive. Higher is better for me."""
    my_body, my_health, my_mask = me
    opp_body, opp_health, opp_mask = opp
    W = WEIGHTS
    bit = bd.bit

    # Free squares: not blocked by a body (tails move away unless they just grew).
    free = bd.full & ~(
        blocking(my_body, my_mask, bit) | blocking(opp_body, opp_mask, bit)
    )

    # Two BFS's at once, one layer (= one distance) per loop, using bitmasks.
    # A square is mine if I reach it while the opponent has not reached it yet
    # (it is still in opp_left), and the other way round. The heads are not free
    # squares, so neither head is counted; that leaves mine - theirs unchanged.
    my_head, opp_head = my_body[0], opp_body[0]
    my_front, opp_front = bit[my_head], bit[opp_head]
    my_left = opp_left = free  # free squares each side has not reached yet
    s = bd.stride
    mine = theirs = 0
    food_mine = food_any = None  # distance to nearest food I reach first / any food
    dist = 0
    while my_front or opp_front:
        mine += (my_front & opp_left).bit_count()
        theirs += (opp_front & my_left).bit_count()
        if my_front & food:
            if food_any is None:
                food_any = dist
            if food_mine is None and my_front & food & opp_left:
                food_mine = dist
        # Grow both areas by one step (shifts = right/left/up/down). A bit pushed off
        # the board lands on the spare column or outside it, never on a free square.
        f = my_front
        my_front = ((f << 1) | (f >> 1) | (f << s) | (f >> s)) & my_left
        my_left ^= my_front
        f = opp_front
        opp_front = ((f << 1) | (f >> 1) | (f << s) | (f >> s)) & opp_left
        opp_left ^= opp_front
        dist += 1

    score = (mine - theirs) * W["territory"]
    my_len, opp_len = len(my_body), len(opp_body)

    # Getting boxed into a space smaller than my body is close to death
    # (reachable space = the squares reached plus the head).
    if (free ^ my_left).bit_count() + 1 < my_len:
        score -= W["trapped_me"]
    if (free ^ opp_left).bit_count() + 1 < opp_len:
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


def search_move(state, root_scores=None):
    """Returns (best move number or None, last completed depth, positions searched).
    Debug only: pass a dict as root_scores to also get each root move's exact score.
    That extra work runs after the search has finished, so it can't change the move."""
    start = time.perf_counter()
    deadline = start + TIME_LIMIT_MS * SAFETY / 1000
    clock = time.perf_counter

    board = state["board"]
    bd = board_tables(board["width"], board["height"])
    bit = bd.bit
    food = 0
    for f in board["food"]:
        food |= bit[bd.square(f)]
    you_id = state["you"]["id"]

    def to_snake(s):
        body = tuple(bd.square(p) for p in s["body"])
        mask = 0
        for sq in body:
            mask |= bit[sq]
        return body, s["health"], mask

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
        my_len, opp_len = len(me[0]), len(opp[0])
        blocked = blocking(me[0], me[2], bit) | blocking(opp[0], opp[2], bit)
        opp_moves = (
            {}
        )  # opponent move -> move_snake result, worked out once per position
        best, best_move = float("-inf"), None
        for my_move in ORDER[my_best.get(path, 0)]:
            my_head, new_me, my_crash = move_snake(me, my_move, food, blocked, bd)
            # Min part: the opponent answers with the reply that is worst for me.
            worst, worst_reply = float("inf"), None
            key = path * 4 + my_move
            for opp_move in ORDER[opp_best.get(key, 0)]:
                if clock() > deadline:
                    raise OutOfTime  # stop right away, keep the last completed depth
                positions += 1
                if opp_move not in opp_moves:
                    opp_moves[opp_move] = move_snake(opp, opp_move, food, blocked, bd)
                opp_head, new_opp, opp_crash = opp_moves[opp_move]
                i_die, they_die = my_crash, opp_crash
                if my_head == opp_head:
                    # Head-to-head: the shorter snake dies, equal length both die.
                    # (Both heads are on the same square, so both ate or neither did.)
                    i_die = i_die or my_len <= opp_len
                    they_die = they_die or opp_len <= my_len
                # Earlier wins and later losses are better (depth = plies left).
                if i_die and they_die:
                    score = draw
                elif i_die:
                    score = loss - depth
                elif they_die:
                    score = win + depth
                else:
                    new_food = food & ~(bit[my_head] | bit[opp_head])  # eaten food goes
                    if depth == 1:
                        score = evaluate(new_me, new_opp, new_food, bd)
                    else:
                        score = max_node(
                            new_me,
                            new_opp,
                            new_food,
                            depth - 1,
                            alpha,
                            min(beta, worst),
                            path * 17 + my_move * 4 + opp_move + 1,
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

    def root_score(my_move, depth):
        """Exact search score of one of my moves (full window, so nothing is pruned away)."""
        worst = float("inf")
        for opp_move in range(4):
            new_me, new_opp, i_die, they_die, new_food = play_turn(
                me, opp, my_move, opp_move, food, bd
            )
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
                    new_me,
                    new_opp,
                    new_food,
                    depth - 1,
                    float("-inf"),
                    float("inf"),
                    my_move * 4 + opp_move + 1,
                )
            worst = min(worst, score)
        return worst

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

    if root_scores is not None and completed:
        # Debug only: score every one of my moves at the last completed depth, with a
        # full window (so pruned moves also get an exact score) and no time limit.
        deadline = float("inf")
        for my_move, name in enumerate(MOVE_NAMES):
            root_scores[name] = root_score(my_move, completed)
    return best_move, completed, positions


def debug_search(state, with_scores=True):
    """Debug only (used by replay.py, never by choose_move): run the exact search that
    choose_move runs and report the move, depth reached and each move's score."""
    scores = {} if with_scores else None
    move, depth, positions = search_move(state, scores)
    return {
        "move": MOVE_NAMES[move] if move is not None else None,
        "depth": depth,
        "positions": positions,
        "scores": scores,
    }


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
    print(
        f"brain stats: {moves} moves, average depth {depth:.2f}, "
        f"average {positions:.0f} positions searched per move"
    )
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
            if (
                0 <= x < board["width"]
                and 0 <= y < board["height"]
                and (x, y) not in taken
            ):
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
