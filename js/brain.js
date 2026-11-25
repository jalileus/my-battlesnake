"use strict";
/*
 * Duel brain in JavaScript: a move-for-move port of brain.py (v9 + territory 3).
 *
 * Iterative-deepening minimax with alpha-beta pruning. Each ply: I pick a move, the
 * opponent picks the reply that is worst for me, then both moves are applied together
 * with the standard Battlesnake rules. Positions are scored by time-aware territory
 * (body squares open up when the tail has passed them), food, length and health.
 *
 * Same moves as brain.py at a fixed depth (checked on 11,300 saved positions, see
 * tools/verify.js); it is only much faster, so it searches deeper in the same time.
 *
 * Board: square = y * width + x. Masks (sets of squares) are Int32Arrays with one
 * number per row, one bit per column, so the flood fill grows a whole row at once.
 */

const { performance } = require("perf_hooks");

let TIME_LIMIT_MS = 150; // thinking budget per move, counted from when the request arrived
const SAFETY = 0.85; // use at most this part of the budget, the rest is slack
let MAX_DEPTH = 40;

const MOVE_NAMES = ["up", "down", "left", "right"];
const DX = [0, 0, -1, 1];
const DY = [1, -1, 0, 0];

// Every tunable number of the evaluation, in one place (same values as brain.py).
const WEIGHTS = {
  win: 1000000,
  loss: -1000000,
  draw: -500000,
  territory: 3,
  trapped_me: 5000,
  trapped_opp: 2000,
  hungry: 40,
  food_hungry_base: 10,
  food_shorter: 6,
  no_food_distance: 25,
  starving: 3000,
  length: 160,
  length_max_lead: 4,
  length_max_behind: 5,
  health_divisor: 10,
  h2h_danger_gap: 2,
  h2h_danger: 0,
  hunt: 8,
};

// Move order: try `first` first, then the rest in the normal order.
const ORDER = [
  [0, 1, 2, 3],
  [1, 0, 2, 3],
  [2, 0, 1, 3],
  [3, 0, 1, 2],
];

const OUT_OF_TIME = new Error("out of time");

// ---------------------------------------------------------------- board tables

const boards = new Map();

function boardTables(w, h) {
  const key = w * 1000 + h;
  let bd = boards.get(key);
  if (bd) return bd;
  const n = w * h;
  const xOf = new Int16Array(n);
  const yOf = new Int16Array(n);
  const step = new Int16Array(n * 4); // step[sq * 4 + move] = next square, or -1 for a wall
  for (let sq = 0; sq < n; sq++) {
    const x = sq % w, y = (sq / w) | 0;
    xOf[sq] = x;
    yOf[sq] = y;
    for (let m = 0; m < 4; m++) {
      const nx = x + DX[m], ny = y + DY[m];
      step[sq * 4 + m] = nx >= 0 && nx < w && ny >= 0 && ny < h ? ny * w + nx : -1;
    }
  }
  bd = { w, h, n, xOf, yOf, step, rowFull: (1 << w) - 1 };
  boards.set(key, bd);
  return bd;
}

function popcount(v) {
  v = v - ((v >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

function hasBit(mask, bd, sq) {
  return (mask[bd.yOf[sq]] >> bd.xOf[sq]) & 1;
}

// ---------------------------------------------------------------- rules

// A snake is { body: array of squares (head first), health }.

/** Add the squares this snake still blocks next turn to `mask`: its body minus the
 * tail (the tail moves away, unless the snake just ate and it is doubled up). */
function addBlocking(mask, body, bd) {
  const n = body.length;
  const last = n > 1 ? n - 1 : n; // body[0 .. n-2], or the whole body if it is 1 long
  for (let i = 0; i < last; i++) {
    const sq = body[i];
    mask[bd.yOf[sq]] |= 1 << bd.xOf[sq];
  }
}

/** One snake's own move: [new head, snake after the move or null, crashed?].
 * `blocked` = squares both bodies block. Head-to-head is checked by the caller. */
function moveSnake(snake, move, food, blocked, bd) {
  const body = snake.body;
  const head = bd.step[body[0] * 4 + move];
  if (head < 0) return [head, null, true]; // hit a wall
  const ate = hasBit(food, bd, head);
  const health = ate ? 100 : snake.health - 1;
  if (health <= 0 || hasBit(blocked, bd, head)) return [head, null, true]; // starved or hit a body
  const n = body.length;
  const newBody = new Array(ate ? n + 1 : n);
  newBody[0] = head;
  for (let i = 1; i < n; i++) newBody[i] = body[i - 1];
  if (ate) newBody[n] = newBody[n - 1]; // grow: the tail doubles up
  return [head, { body: newBody, health }, false];
}

// ---------------------------------------------------------------- evaluation

// Scratch masks for evaluate (it never runs twice at once).
let ev = null;
function scratch(h) {
  if (!ev || ev.h !== h) {
    const a = () => new Int32Array(h);
    ev = { h, free: a(), myLeft: a(), oppLeft: a(), mySeen: a(), oppSeen: a(), myFront: a(), oppFront: a() };
  }
  return ev;
}

/** Score a position where both snakes are alive. Higher is better for me. */
function evaluate(me, opp, food, bd) {
  const W = WEIGHTS;
  const h = bd.h, full = bd.rowFull;
  const myBody = me.body, oppBody = opp.body;
  const s = scratch(h);
  const { free, myLeft, oppLeft, mySeen, oppSeen, myFront, oppFront } = s;

  // Free squares: not blocked by a body (tails move away unless they just grew).
  free.fill(0);
  addBlocking(free, myBody, bd);
  addBlocking(free, oppBody, bd);
  for (let r = 0; r < h; r++) {
    free[r] = full & ~free[r];
    myLeft[r] = free[r];
    oppLeft[r] = free[r];
    mySeen[r] = 0;
    oppSeen[r] = 0;
  }
  const myHead = myBody[0], oppHead = oppBody[0];
  mySeen[bd.yOf[myHead]] = 1 << bd.xOf[myHead];
  oppSeen[bd.yOf[oppHead]] |= 1 << bd.xOf[oppHead];
  myFront.set(mySeen);
  oppFront.set(oppSeen);

  // Two flood fills at once, one layer (= one turn) per loop. A square is mine if I
  // reach it while the opponent has not reached it yet, and the other way round.
  let mine = 0, theirs = 0;
  let foodMine = -1, foodAny = -1; // distance to nearest food I reach first / any food
  const nMe = myBody.length, nOpp = oppBody.length;
  const last = Math.max(nMe, nOpp); // by then both bodies have fully pulled back
  let dist = 0;
  let anyFront = true;
  while (anyFront || dist < last) {
    let foodHere = false, foodFirst = false;
    for (let r = 0; r < h; r++) {
      const mf = myFront[r], of = oppFront[r];
      if (mf) {
        mine += popcount(mf & oppLeft[r]);
        const fm = mf & food[r];
        if (fm) {
          foodHere = true;
          if (fm & oppLeft[r]) foodFirst = true;
        }
      }
      if (of) theirs += popcount(of & myLeft[r]);
    }
    if (foodHere) {
      if (foodAny < 0) foodAny = dist;
      if (foodMine < 0 && foodFirst) foodMine = dist;
    }
    dist += 1; // the squares found below are reached on turn `dist`
    // Bodies pull back: segment i of a length-n snake is gone after n - i turns, so on
    // turn `dist` segment n - dist opens up (turn 1, the tail, is already free unless
    // it just grew; a grown tail sits on segment n - 2 too).
    if (dist >= 2) {
      if (nMe - dist >= 1) {
        const sq = myBody[nMe - dist], r = bd.yOf[sq], b = 1 << bd.xOf[sq];
        free[r] |= b; myLeft[r] |= b; oppLeft[r] |= b;
      }
      if (nOpp - dist >= 1) {
        const sq = oppBody[nOpp - dist], r = bd.yOf[sq], b = 1 << bd.xOf[sq];
        free[r] |= b; myLeft[r] |= b; oppLeft[r] |= b;
      }
    }
    // Grow both areas by one step from everything reached so far, so a square that
    // opens next to old area is found too.
    anyFront = false;
    for (let r = 0; r < h; r++) {
      let g = mySeen[r];
      let nb = (g << 1) | (g >> 1);
      if (r > 0) nb |= mySeen[r - 1];
      if (r < h - 1) nb |= mySeen[r + 1];
      myFront[r] = nb & myLeft[r];
      g = oppSeen[r];
      nb = (g << 1) | (g >> 1);
      if (r > 0) nb |= oppSeen[r - 1];
      if (r < h - 1) nb |= oppSeen[r + 1];
      oppFront[r] = nb & oppLeft[r];
    }
    for (let r = 0; r < h; r++) {
      const mf = myFront[r], of = oppFront[r];
      if (mf | of) anyFront = true;
      myLeft[r] ^= mf;
      mySeen[r] |= mf;
      oppLeft[r] ^= of;
      oppSeen[r] |= of;
    }
  }

  let score = (mine - theirs) * W.territory;
  const myLen = nMe, oppLen = nOpp;

  // Getting boxed into a space smaller than my body is close to death
  // (reachable space = the squares reached plus the head).
  let myReach = 0, oppReach = 0;
  for (let r = 0; r < h; r++) {
    myReach += popcount(free[r] ^ myLeft[r]);
    oppReach += popcount(free[r] ^ oppLeft[r]);
  }
  if (myReach + 1 < myLen) score -= W.trapped_me;
  if (oppReach + 1 < oppLen) score += W.trapped_opp;

  // Food: only chase it when hungry or shorter than the opponent.
  const nearest = foodMine >= 0 ? foodMine : foodAny;
  const myHealth = me.health;
  let weight;
  if (myHealth < W.hungry) weight = W.food_hungry_base + (W.hungry - myHealth);
  else if (myLen < oppLen) weight = W.food_shorter;
  else weight = 0;
  score -= (nearest >= 0 ? nearest : W.no_food_distance) * weight;
  if (nearest < 0 || nearest > myHealth) score -= W.starving; // cannot reach food in time

  // Length: reward growing until I am a bit longer.
  const lead = Math.max(-W.length_max_behind, Math.min(myLen - oppLen, W.length_max_lead));
  score += lead * W.length;

  // Small health term (floor division, like Python's //).
  score += Math.floor((myHealth - opp.health) / W.health_divisor);

  // Head-to-head: stay away from an equal or longer snake, or hunt a shorter one.
  const headGap = Math.abs(bd.xOf[myHead] - bd.xOf[oppHead]) + Math.abs(bd.yOf[myHead] - bd.yOf[oppHead]);
  if (oppLen >= myLen && headGap <= W.h2h_danger_gap) score -= W.h2h_danger;
  else if (myLen > oppLen) score -= headGap * W.hunt;
  return score;
}

// ---------------------------------------------------------------- search

const PATH_LIMIT = 2 ** 48;
const PATH_MOD = 2 ** 48 - 59; // keeps path keys exact numbers on very deep searches

/** Returns [best move number or null, last completed depth, positions searched].
 * `startMs` = when the request arrived (performance.now()), so time spent waiting
 * behind another request counts against this move's budget. */
function searchMove(state, startMs) {
  const start = startMs === undefined ? performance.now() : startMs;
  const deadline = start + TIME_LIMIT_MS * SAFETY;

  const board = state.board;
  const bd = boardTables(board.width, board.height);
  const h = bd.h;
  const sqOf = (p) => p.y * bd.w + p.x;
  let food = new Int32Array(h);
  for (const f of board.food) food[f.y] |= 1 << f.x;
  const youId = state.you.id;
  const toSnake = (s) => ({ body: s.body.map(sqOf), health: s.health });
  const me0 = toSnake(state.you);
  const opp0 = toSnake(board.snakes.find((s) => s.id !== youId));

  const win = WEIGHTS.win, loss = WEIGHTS.loss, draw = WEIGHTS.draw;
  // Best moves found by the previous depth, per position in the tree (move ordering).
  // A path (the moves from the root) is one number: path * 17 + (1..16).
  const myBest = new Map(); // path -> my best move
  const oppBest = new Map(); // path * 4 + my move -> opponent's strongest reply
  let positions = 0;
  let rootFound = null;

  function maxNode(me, opp, food, depth, alpha, beta, path) {
    const myLen = me.body.length, oppLen = opp.body.length;
    const blocked = new Int32Array(h);
    addBlocking(blocked, me.body, bd);
    addBlocking(blocked, opp.body, bd);
    const oppMoves = [null, null, null, null]; // worked out once per position
    let best = -Infinity, bestMove = null;
    const order = ORDER[myBest.get(path) || 0];
    const childBase = (path > PATH_LIMIT ? path % PATH_MOD : path) * 17;
    for (let i = 0; i < 4; i++) {
      const myMove = order[i];
      const [myHead, newMe, myCrash] = moveSnake(me, myMove, food, blocked, bd);
      // Min part: the opponent answers with the reply that is worst for me.
      let worst = Infinity, worstReply = null;
      const key = path * 4 + myMove;
      const oppOrder = ORDER[oppBest.get(key) || 0];
      for (let j = 0; j < 4; j++) {
        const oppMove = oppOrder[j];
        if (performance.now() > deadline) throw OUT_OF_TIME;
        positions++;
        let om = oppMoves[oppMove];
        if (om === null) om = oppMoves[oppMove] = moveSnake(opp, oppMove, food, blocked, bd);
        const oppHead = om[0], newOpp = om[1];
        let iDie = myCrash, theyDie = om[2];
        if (myHead === oppHead) {
          // Head-to-head: the shorter snake dies, equal length both die.
          iDie = iDie || myLen <= oppLen;
          theyDie = theyDie || oppLen <= myLen;
        }
        let score;
        if (iDie && theyDie) score = draw;
        else if (iDie) score = loss - depth;
        else if (theyDie) score = win + depth;
        else {
          let newFood = food;
          if (hasBit(food, bd, myHead) || hasBit(food, bd, oppHead)) {
            newFood = food.slice(); // eaten food goes
            newFood[bd.yOf[myHead]] &= ~(1 << bd.xOf[myHead]);
            newFood[bd.yOf[oppHead]] &= ~(1 << bd.xOf[oppHead]);
          }
          if (depth === 1) score = evaluate(newMe, newOpp, newFood, bd);
          else
            score = maxNode(newMe, newOpp, newFood, depth - 1, alpha, Math.min(beta, worst),
              childBase + myMove * 4 + oppMove + 1);
        }
        if (score < worst) {
          worst = score;
          worstReply = oppMove;
        }
        if (worst <= alpha) break; // this move can't beat one I already have
      }
      oppBest.set(key, worstReply);

      if (worst > best) {
        best = worst;
        bestMove = myMove;
        if (path === 0) rootFound = myMove; // best fully searched root move so far
      }
      alpha = Math.max(alpha, best);
      if (alpha >= beta) break; // the opponent would never let us reach this position
    }
    myBest.set(path, bestMove);
    return best;
  }

  let bestMove = null, completed = 0;
  for (let depth = 1; depth <= MAX_DEPTH; depth++) {
    rootFound = null;
    let score;
    try {
      score = maxNode(me0, opp0, food, depth, -Infinity, Infinity, 0);
    } catch (e) {
      if (e !== OUT_OF_TIME) throw e;
      // The unfinished depth searched the old best move first. If another move was
      // fully searched after it and scored higher, play it.
      if (completed && rootFound !== null) bestMove = rootFound;
      break;
    }
    bestMove = myBest.get(0);
    completed = depth;
    if (Math.abs(score) >= win - 1000) break; // forced win or loss found
  }
  return [bestMove, completed, positions];
}

// ---------------------------------------------------------------- fallback + entry point

/** Simple safe move: the free neighbour square with the most room behind it. */
function fallbackMove(state) {
  try {
    const board = state.board, w = board.width, hh = board.height;
    const taken = new Set();
    for (const s of board.snakes) for (let i = 0; i < s.body.length - 1; i++) taken.add(s.body[i].y * w + s.body[i].x);
    const head = state.you.head;
    let best = "up", bestRoom = -1;
    for (let m = 0; m < 4; m++) {
      const x = head.x + DX[m], y = head.y + DY[m];
      if (x < 0 || x >= w || y < 0 || y >= hh || taken.has(y * w + x)) continue;
      const seen = new Set([y * w + x]), queue = [[x, y]];
      while (queue.length && seen.size < 200) {
        const [cx, cy] = queue.pop();
        for (let k = 0; k < 4; k++) {
          const nx = cx + DX[k], ny = cy + DY[k], sq = ny * w + nx;
          if (nx >= 0 && nx < w && ny >= 0 && ny < hh && !taken.has(sq) && !seen.has(sq)) {
            seen.add(sq);
            queue.push([nx, ny]);
          }
        }
      }
      if (seen.size > bestRoom) {
        bestRoom = seen.size;
        best = MOVE_NAMES[m];
      }
    }
    return best;
  } catch (e) {
    return "up";
  }
}

// Debug counters: average depth per game, printed by the server at the end of a game.
const STATS = new Map(); // game id -> [moves, total depth, total positions]

function chooseMove(state, startMs) {
  try {
    if (state.board.snakes.length !== 2) return { move: fallbackMove(state) };
    const [move, depth, positions] = searchMove(state, startMs);
    const id = state.game && state.game.id;
    if (id) {
      if (!STATS.has(id) && STATS.size >= 200) STATS.delete(STATS.keys().next().value);
      const s = STATS.get(id) || [0, 0, 0];
      s[0]++; s[1] += depth; s[2] += positions;
      STATS.set(id, s);
    }
    return { move: move !== null ? MOVE_NAMES[move] : fallbackMove(state), depth, positions };
  } catch (e) {
    return { move: fallbackMove(state) };
  }
}

function gameStats(id, reset = true) {
  const s = STATS.get(id);
  if (reset) STATS.delete(id);
  return s ? { moves: s[0], avgDepth: s[1] / s[0], avgPositions: s[2] / s[0] } : null;
}

module.exports = {
  WEIGHTS,
  MOVE_NAMES,
  searchMove,
  chooseMove,
  fallbackMove,
  evaluate,
  gameStats,
  setTimeLimit: (ms) => { TIME_LIMIT_MS = ms; },
  getTimeLimit: () => TIME_LIMIT_MS,
  setMaxDepth: (d) => { MAX_DEPTH = d; },
};
