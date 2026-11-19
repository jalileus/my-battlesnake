"""Local Battlesnake Duel simulator (1v1, 11x11, standard rules).

Usage:
    python simulate.py brain_a brain_b 100

Each brain is a Python file in this folder with a choose_move(state) function.
"brain_v1" also finds "brain-v1.py" (a dash is not allowed in import names).
"""

import importlib.util
import os
import random
import sys
import time

WIDTH = HEIGHT = 11
START_LENGTH = 3
MAX_HEALTH = 100
FOOD_SPAWN_CHANCE = 15  # percent per turn, standard mode
MINIMUM_FOOD = 1
MAX_TURNS = 2000  # safety cap so a game can never run forever (counts as a draw)
MOVES = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- loading brains

def load_brain(name):
    """Import choose_move from <name>.py (also tries '-' instead of '_')."""
    name = name[:-3] if name.endswith(".py") else name
    for candidate in (name, name.replace("_", "-"), name.replace("-", "_")):
        path = os.path.join(HERE, candidate + ".py")
        if os.path.isfile(path):
            # A unique module name lets you load the same file twice (brain vs brain).
            spec = importlib.util.spec_from_file_location(
                "sim_" + candidate.replace("-", "_") + "_" + str(id(path)), path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.choose_move
    sys.exit(f"Could not find {name}.py in {HERE}")


# ---------------------------------------------------------------- game setup

def place_snakes(rng):
    """Standard fixed start spots: both snakes on corners, or both on edge middles."""
    lo, mid, hi = 1, (WIDTH - 1) // 2, WIDTH - 2
    corners = [(lo, lo), (lo, hi), (hi, lo), (hi, hi)]
    cardinals = [(lo, mid), (mid, lo), (mid, hi), (hi, mid)]
    spots = corners if rng.random() < 0.5 else cardinals
    rng.shuffle(spots)
    return spots[:2]


def place_start_food(heads, rng):
    """One food diagonal to each snake (not toward the center), plus one in the center."""
    cx, cy = (WIDTH - 1) // 2, (HEIGHT - 1) // 2
    food = []
    for hx, hy in heads:
        options = []
        for fx, fy in [(hx - 1, hy - 1), (hx - 1, hy + 1), (hx + 1, hy - 1), (hx + 1, hy + 1)]:
            if (fx, fy) == (cx, cy) or (fx, fy) in food:
                continue
            away_from_center = (
                (fx < hx < cx) or (cx < hx < fx) or (fy < hy < cy) or (cy < hy < fy)
            )
            if not away_from_center:
                options.append((fx, fy))
        if options:
            food.append(rng.choice(options))
    if (cx, cy) not in food and (cx, cy) not in heads:
        food.append((cx, cy))
    return food


def spawn_food(snakes, food, rng):
    """Standard food spawning: keep at least MINIMUM_FOOD, else 15% chance of one more."""
    if len(food) < MINIMUM_FOOD:
        count = MINIMUM_FOOD - len(food)
    elif rng.randrange(100) < FOOD_SPAWN_CHANCE:
        count = 1
    else:
        return
    taken = set(food)
    for s in snakes:
        taken.update(s["body"])
    free = [(x, y) for x in range(WIDTH) for y in range(HEIGHT) if (x, y) not in taken]
    rng.shuffle(free)
    food.extend(free[:count])


# ---------------------------------------------------------------- API game-state format

def point(p):
    return {"x": p[0], "y": p[1]}


def snake_json(s):
    return {
        "id": s["id"],
        "name": s["name"],
        "health": s["health"],
        "body": [point(p) for p in s["body"]],
        "latency": "0",
        "head": point(s["body"][0]),
        "length": len(s["body"]),
        "shout": "",
        "squad": "",
        "customizations": {"color": "#888888", "head": "default", "tail": "default"},
    }


def build_state(game_id, turn, snakes, food, you):
    """Build the dict exactly like the real Battlesnake /move request."""
    alive = [snake_json(s) for s in snakes if s["alive"]]
    return {
        "game": {
            "id": game_id,
            "ruleset": {
                "name": "standard",
                "version": "v1.2.3",
                "settings": {
                    "foodSpawnChance": FOOD_SPAWN_CHANCE,
                    "minimumFood": MINIMUM_FOOD,
                    "hazardDamagePerTurn": 0,
                    "hazardMap": "",
                    "hazardMapAuthor": "",
                    "royale": {"shrinkEveryNTurns": 0},
                    "squad": {
                        "allowBodyCollisions": False,
                        "sharedElimination": False,
                        "sharedHealth": False,
                        "sharedLength": False,
                    },
                },
            },
            "map": "standard",
            "timeout": 500,
            "source": "custom",
        },
        "turn": turn,
        "board": {
            "height": HEIGHT,
            "width": WIDTH,
            "food": [point(f) for f in food],
            "hazards": [],
            "snakes": alive,
        },
        "you": snake_json(you),
    }


# ---------------------------------------------------------------- one game

def play_game(game_id, brains, names, setup_rng, rng, timings):
    """Play one game. brains[i] controls snake i. Returns winner index, or None for a draw."""
    starts = place_snakes(setup_rng)
    snakes = [
        {"id": f"snake-{i}", "name": names[i], "health": MAX_HEALTH,
         "body": [starts[i]] * START_LENGTH, "alive": True}
        for i in range(2)
    ]
    food = place_start_food(starts, setup_rng)

    for turn in range(MAX_TURNS):
        # 1. Ask every live snake for its move (all moves are simultaneous).
        moves = {}
        for i, s in enumerate(snakes):
            state = build_state(game_id, turn, snakes, food, s)
            start = time.perf_counter()
            try:
                move = brains[i](state)
            except Exception as e:  # a crashing snake loses
                print(f"  [game {game_id}] {names[i]} crashed on turn {turn}: {e!r}")
                move = None
            timings.setdefault(names[i], []).append((time.perf_counter() - start) * 1000)
            if move not in MOVES:
                if move is not None:
                    print(f"  [game {game_id}] {names[i]} returned invalid move {move!r}")
                s["alive"] = False
            moves[i] = move

        # 2. Move: new head in front, tail removed. 3. Lose 1 health.
        for i, s in enumerate(snakes):
            if s["alive"]:
                dx, dy = MOVES[moves[i]]
                hx, hy = s["body"][0]
                s["body"] = [(hx + dx, hy + dy)] + s["body"][:-1]
                s["health"] -= 1

        # 4. Eat: health back to 100, grow by repeating the tail segment.
        eaten = set()
        for s in snakes:
            if s["alive"] and s["body"][0] in food:
                s["health"] = MAX_HEALTH
                s["body"].append(s["body"][-1])
                eaten.add(s["body"][0])
        food = [f for f in food if f not in eaten]

        # 5. Spawn new food.
        spawn_food([s for s in snakes if s["alive"]], food, rng)

        # 6. Eliminations. First starvation and walls...
        for s in snakes:
            if not s["alive"]:
                continue
            x, y = s["body"][0]
            if s["health"] <= 0 or not (0 <= x < WIDTH and 0 <= y < HEIGHT):
                s["alive"] = False

        # ...then collisions, all decided at the same time.
        dead = set()
        live = [s for s in snakes if s["alive"]]
        for s in live:
            head = s["body"][0]
            for other in live:
                if head in other["body"][1:]:  # hit a body (own or other)
                    dead.add(s["id"])
                elif other is not s and head == other["body"][0]:
                    if len(s["body"]) <= len(other["body"]):  # head-to-head: shorter (or equal) dies
                        dead.add(s["id"])
        for s in snakes:
            if s["id"] in dead:
                s["alive"] = False

        alive = [i for i, s in enumerate(snakes) if s["alive"]]
        if len(alive) == 1:
            return alive[0]
        if not alive:
            return None
    return None  # hit MAX_TURNS: draw


# ---------------------------------------------------------------- main

def main():
    if len(sys.argv) != 4:
        sys.exit("Usage: python simulate.py brain_a brain_b number_of_games")
    name_a, name_b, n_games = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if name_a == name_b:  # keep the two sides apart in the results
        label_a, label_b = name_a + " (A)", name_b + " (B)"
    else:
        label_a, label_b = name_a, name_b

    sys.path.insert(0, HERE)  # so brains can import their own helper files
    brain_a, brain_b = load_brain(name_a), load_brain(name_b)

    wins = {label_a: 0, label_b: 0}
    draws = 0
    timings = {}
    rng = random.Random()
    base_seed = random.randrange(10**9)

    for g in range(n_games):
        # Swap sides every game. Each pair of games uses the same start layout,
        # so both brains get to play from both spots.
        if g % 2 == 0:
            brains, names = [brain_a, brain_b], [label_a, label_b]
        else:
            brains, names = [brain_b, brain_a], [label_b, label_a]
        setup_rng = random.Random(base_seed + g // 2)
        winner = play_game(f"sim-{g}", brains, names, setup_rng, rng, timings)
        if winner is None:
            draws += 1
        else:
            wins[names[winner]] += 1

    print(f"\nGames played: {n_games}")
    for label in (label_a, label_b):
        t = timings.get(label, [0])
        print(f"{label:>20}: {wins[label]:4d} wins   "
              f"avg {sum(t) / len(t):7.2f} ms/move   max {max(t):7.2f} ms/move")
    print(f"{'draws':>20}: {draws:4d}")


if __name__ == "__main__":
    main()
