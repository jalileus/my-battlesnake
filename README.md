# Cassiopeia: a Battlesnake for Duels

### Author: Alhajali Jalal (Алхажали Жалал)
### Group: M05-607в

Cassiopeia is a [Battlesnake](https://play.battlesnake.com) built for **Duels**: one
against one, on an 11x11 board. It decides each move with an **iterative-deepening
minimax search with alpha-beta pruning**, and it scores positions mainly by
**Voronoi territory** (which squares each snake can reach first).

It was written for a university course assignment.

## Run it locally

You need **Python 3.10 or newer** (the code uses `int.bit_count`; tested with 3.11 and 3.12) and the packages in `requirements.txt` (only Flask).

**Windows** (PowerShell, after installing Python from [python.org](https://www.python.org/downloads/)):

```powershell
python -m pip install -r requirements.txt
python main.py
```

**Mac** (Terminal, after installing Python from python.org or with Homebrew):

```sh
python3 -m pip install -r requirements.txt
python3 main.py
```

The server starts on port 8000 (set the `PORT` environment variable to use another
port). Open <http://localhost:8000> in a browser and you should see the snake's info:

```json
{"apiversion": "1", "author": "jalileus", "color": "#f13232", "head": "Evil", "tail": "Bolt", "version": "v9"}
```

Battlesnake sends the game to `/start`, `/move` and `/end`. There is also a debug
page, <http://localhost:8000/bench>, which runs one search on a saved position
(`bench_state.json.json`) and shows the depth reached, positions searched, time
taken and the move chosen.

## How it works

**The web server.** `main.py` and `server.py` are the web server (Flask). They
answer the Battlesnake API and pass each `/move` request to `choose_move` in
`brain.py`. At the end of a game, the server prints the average search depth for
that game.

**The search.** `brain.py` holds `choose_move(state)`. On every turn it searches
1 move ahead, then 2, then 3, and so on, until the time runs out (iterative
deepening). It always tries the next depth, and when time runs out it plays the
best move found: the last completed depth's move, or a better root move that the
unfinished depth already proved. In each ply, the snake picks a move and the opponent picks
the reply that is worst for me (minimax). Branches that cannot change the result are skipped
(alpha-beta pruning), and the best moves from the previous depth are tried first,
which makes that pruning work better. If the board does not have exactly two
snakes, or anything goes wrong, it falls back to the simpler `brain_v2` logic.

**The time limit.** `TIME_LIMIT_MS` at the top of `brain.py` (now 130) is the
thinking budget per move. The search uses at most 85% of it. Everything runs on
**one CPU core** and only uses the Python standard library, so a move is answered well
under Battlesnake's 500 ms limit.

**The evaluation.** When the search stops, positions are scored by the
`WEIGHTS` dictionary in `brain.py`: territory (squares I reach before the
opponent), being trapped in a space smaller than my body, food (only when hungry
or shorter), length lead, health, and head-to-head danger or hunting. A win or a
loss gets a huge score. The territory count is time-aware: each body square opens
up on the turn the tail really leaves it, so bodies that pull back are not counted
as walls. All the tunable numbers live in `WEIGHTS`, so they can be
changed in one place.

**Speed.** Squares are numbers and the board is stored as bitmasks (one bit per
square in a Python integer), so the territory flood fill grows a whole layer at
once with a few bit shifts.

## Tools

All tools play games that implement the standard Battlesnake rules (our own implementation, not the official engine).
The names `brain`, `brain_v6`, and so on are the `.py` files in this folder.

**`simulate.py`**: play games between two brains, one at a time, and print wins,
draws and time per move.

```sh
python simulate.py brain_v6 brain 100
```

**`simulate_fast.py`**: the same, but many games in parallel on several CPU cores.
`--time` sets the thinking time for both snakes, `--time-a` / `--time-b` for each
one separately, and `--procs` the number of processes. It also prints the win
rate of the second brain with a 95% confidence range.

```sh
python simulate_fast.py brain_v6 brain 150 --time 100
python simulate_fast.py brain_v6 brain 150 --time-a 100 --time-b 130
```

**`tune.py`**: try several values for one number in `WEIGHTS` (in memory, the file
is not changed) against an opponent brain, and print the win rate for each value.

```sh
python tune.py territory 4 6 8 --games 200 --opponent brain_v6
```

**`replay.py`**: replay saved `/move` requests (JSON files) through `brain.py` to
see why a game was lost. For each turn it prints the board, the score of every
move, and what a much deeper search would choose. The default folder is
`lost_games_turns`.

```sh
python replay.py
python replay.py lost_games_turns
```

## Version history

`brain.py` is the version that plays. Each `brain_vN.py` is a frozen snapshot.

| Version | What changed |
|---|---|
| `brain_v1.py` | Safe moves only (no walls or bodies), plus a flood fill to avoid small spaces |
| `brain_v2.py` | One move of lookahead against the opponent's worst reply, Voronoi territory |
| `brain_v3.py` | Iterative-deepening minimax search with alpha-beta pruning |
| `brain_v4.py` | Weight tuning: `length` and `h2h_danger` |
| `brain_v5.py` | More tuning: `territory` and `length_max_lead` |
| `brain_v6.py` | Bitboard speed-up: same moves, about 4.5x faster search |
| `brain_v7.py` | Further speed-up (same moves as v6, about 1.45x faster) and a 130 ms time limit |
| `brain_v9.py` | Time-aware territory: the flood fill opens each body square on the turn the tail really leaves it, so tails and bodies that pull back are not counted as walls. Also keeps the best move found by an unfinished deeper search, and always tries the next depth until the time runs out. |

## Live deployment

The snake is hosted on [Render](https://render.com) (free plan) at
<https://cassiopeia-wjj6.onrender.com/> and is registered on
[arena.battlesnake.com](https://arena.battlesnake.com) as **cassiopeia**.

Free Render services go to sleep after a while without traffic, so the first
request after a quiet period can be slow.

## About the code

The server skeleton (`main.py` / `server.py`) comes from the official Battlesnake
starter template for Python. The search, evaluation, simulator and tools are
original and were written with AI assistance ("vibe coding"), as the assignment
allows. No code from other players' open-source snakes was copied.
