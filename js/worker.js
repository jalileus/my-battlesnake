"use strict";
// One search thread. The server sends {id, state, arrivedEpoch, timeLimit}; the deadline
// is counted from arrivedEpoch (when the request reached the server), so time spent
// waiting for a free thread counts against the move's budget.
const { parentPort } = require("worker_threads");
const { performance } = require("perf_hooks");
const brain = require("./brain");

// Warm-up: a few short searches at start, so the first real move is already fast
// (Node compiles the hot code while it runs).
try {
  const fs = require("fs");
  const path = require("path");
  const file = path.join(__dirname, "bench_state.json");
  if (fs.existsSync(file)) {
    const state = JSON.parse(fs.readFileSync(file, "utf8"));
    const keep = brain.getTimeLimit();
    brain.setTimeLimit(100);
    for (let i = 0; i < 5; i++) brain.searchMove(state);
    brain.setTimeLimit(keep);
  }
} catch (e) {
  // warm-up is optional
}

parentPort.on("message", ({ id, state, arrivedEpoch, timeLimit }) => {
  brain.setTimeLimit(timeLimit);
  const startMs = arrivedEpoch - performance.timeOrigin; // same moment, this thread's clock
  const r = brain.chooseMove(state, startMs);
  parentPort.postMessage({ id, move: r.move, depth: r.depth || 0, positions: r.positions || 0 });
});
