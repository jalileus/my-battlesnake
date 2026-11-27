# Cassiopeia: a Battlesnake for Duels

### Author: Alhajali Jalal (Алхажали Жалал)
### Group: M05-607в

Cassiopeia is a [Battlesnake](https://play.battlesnake.com) built for **Duels**: one
against one, on an 11x11 board. It decides each move with an **iterative-deepening
minimax search with alpha-beta pruning**, and it scores positions mainly by
**Voronoi territory** (which squares each snake can reach first).

The snake that plays is the **JavaScript version in `js/`** (Node.js). It is a
move-for-move port of the Python version `brain.py`: at the same search depth both
pick exactly the same moves, but the JavaScript one is about 6x faster, so it
searches about 2 moves deeper in the same time. The Python version is kept as the
reference and is used by the testing tools.

It was written for a university course assignment.

## Run it locally

### The JavaScript snake (the one that plays)

You need **Node.js 18 or newer** ([nodejs.org](https://nodejs.org)). There are no
packages to install: the server only uses Node's built-in modules.

**Windows** (PowerShell) or **Mac** (Terminal), from this folder:

```sh
node js/server.js
```

The server starts on port 8000 (set the `PORT` environment variable to use another
port). Open <http://localhost:8000> in a browser and you should see the snake's info:

```json
{"apiversion": "1", "author": "jalileus", "color": "#f13232", "head": "Evil", "tail": "Bolt", "version": "v10-js"}
```

Battlesnake sends the game to `/start`, `/move` and `/end`. There is also a debug
page, <http://localhost:8000/bench>, which runs one search on a saved position
(`js/bench_state.json`) and shows the depth reached, positions searched, time
taken and the move chosen.

### The Python version (reference and tools)

`brain.py` is the Python version the JavaScript snake was ported from. The testing
tools below use it. They need **Python 3.10 or newer** (the code uses
`int.bit_count`) and no extra packages.

The old Python web server (`main.py` / `server.py`, Flask) and the frozen versions
are kept in `old_python/` and are no longer used.

## How it works

**The web server.** `js/server.js` answers the Battlesnake API. Each `/move`
request is searched in a separate worker thread (`js/worker.js`, 3 threads by
default), so when several games ask for a move at the same time, each one still
gets its answer within its own time limit. The time limit is counted from the
moment the request arrives. At the end of a game, the server prints the average
search depth for that game.

**The search.** `js/brain.js` holds `chooseMove(state)`. On every turn it searches
1 move ahead, then 2, then 3, and so on, until the time runs out (iterative
deepening). It always tries the next depth, and when time runs out it plays the
best move found: the last completed depth's move, or a better root move that the
unfinished depth already proved. In each ply, the snake picks a move and the
opponent picks the reply that is worst for it (minimax). Branches that cannot
change the result are skipped (alpha-beta pruning), and the best moves from the
previous depth are tried first, which makes that pruning work better. If the board
does not have exactly two snakes, or anything goes wrong, it plays the free move
with the most room behind it.

**The time limit.** `TIME_LIMIT_MS` (now 220, or set the `TIME_LIMIT_MS`
environment variable) is the thinking budget per move. The search uses at most 85%
of it, so a move is answered well under Battlesnake's 500 ms limit, also on one
CPU core.

**The evaluation.** When the search stops, positions are scored by the `WEIGHTS`
table: territory (squares the snake reaches before the opponent), being trapped in
a space smaller than its body, food (only when hungry or shorter), length lead,
health, and head-to-head danger or hunting. A win or a loss gets a huge score. The
territory count is time-aware: each body square opens up on the turn the tail
really leaves it, so bodies that pull back are not counted as walls. All the
tunable numbers live in `WEIGHTS`, so they can be changed in one place.

**Speed.** Squares are numbers and the board is stored as bitmasks (in JavaScript
one number per row, one bit per column), so the territory flood fill grows a whole
layer at once with a few bit shifts.

## Tools

All tools play games that implement the standard Battlesnake rules (our own
implementation, not the official engine). The names `brain`, `brain_v6`, and so on
are `.py` files (old versions are in `old_python/`, for example `old_python/brain_v6`);
`candidates/brain_js` plays the JavaScript snake (it needs Node.js).

**`simulate.py`**: play games between two brains, one at a time, and print wins,
draws and time per move.

```sh
python simulate.py old_python/brain_v6 brain 100
```

**`simulate_fast.py`**: the same, but many games in parallel on several CPU cores.
`--time` sets the thinking time for both snakes, `--time-a` / `--time-b` for each
one separately, and `--procs` the number of processes. It also prints the win
rate of the second brain with a 95% confidence range.

```sh
python simulate_fast.py old_python/brain_v6 brain 150 --time 100
python simulate_fast.py brain candidates/brain_js 100 --time 40
```

**`tune.py`**: try several values for one number in `WEIGHTS` (in memory, the file
is not changed) against an opponent brain, and print the win rate for each value.

```sh
python tune.py territory 4 6 8 --games 200 --opponent old_python/brain_v6
```

**`replay.py`**: replay saved `/move` requests (JSON files) through `brain.py` to
see why a game was lost. For each turn it prints the board, the score of every
move, and what a much deeper search would choose. The default folder is
`lost_games_turns`.

```sh
python replay.py
python replay.py lost_games_turns
```

**JavaScript checks** (in `js/tools/`): `dump_expected.py` records the Python
version's move and number of positions searched at a fixed depth on saved
positions, and `verify.js` checks that the JavaScript version gives exactly the
same. `speed.js` shows the average depth and speed at a time limit.

```sh
python js/tools/dump_expected.py
node js/tools/verify.js
node js/tools/speed.js 220
```

## Version history

The JavaScript version in `js/` is the version that plays. `brain.py` is the
Python version it was ported from. Each `old_python/brain_vN.py` is a frozen snapshot
(`brain_v2.py` stays in the main folder because `brain.py` uses it as a fallback).

| Version | What changed |
|---|---|
| `old_python/brain_v1.py` | Safe moves only (no walls or bodies), plus a flood fill to avoid small spaces |
| `brain_v2.py` | One move of lookahead against the opponent's worst reply, Voronoi territory |
| `old_python/brain_v3.py` | Iterative-deepening minimax search with alpha-beta pruning |
| `old_python/brain_v4.py` | Weight tuning: `length` and `h2h_danger` |
| `old_python/brain_v5.py` | More tuning: `territory` and `length_max_lead` |
| `old_python/brain_v6.py` | Bitboard speed-up: same moves, about 4.5x faster search |
| `old_python/brain_v7.py` | Further speed-up (same moves as v6, about 1.45x faster) and a 130 ms time limit |
| `old_python/brain_v8.py` | Keeps the best move found by an unfinished deeper search, and always tries the next depth until the time runs out |
| `old_python/brain_v9.py` | Time-aware territory: the flood fill opens each body square on the turn the tail really leaves it, so tails and bodies that pull back are not counted as walls |
| `brain.py` | v9 with the `territory` weight lowered from 6 to 3 (better at the shallow depth reached on the server) |
| `js/brain.js` (v10-js) | JavaScript port of `brain.py`: the same moves at a fixed depth (checked on 11,300 saved positions), about 6x faster, so about 2 moves deeper; searches run in worker threads |

## Live deployment

The snake is hosted on [Render](https://render.com) (free plan, Node.js, root
directory `js`, start command `node server.js`) at
<https://cassiopeia-wjj6.onrender.com/> and is registered on
[arena.battlesnake.com](https://arena.battlesnake.com) as **cassiopeia**.

Free Render services go to sleep after a while without traffic, so the first
request after a quiet period can be slow.

## About the code

The old Python server skeleton (`old_python/main.py` / `old_python/server.py`) comes from the official
Battlesnake starter template for Python. The search, evaluation, simulator, tools
and the JavaScript version (`js/`) are original and were written with AI
assistance ("vibe coding"), as the assignment allows. No code from other players'
open-source snakes was copied.
