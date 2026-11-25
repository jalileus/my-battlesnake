// Bridge for the Python simulator: reads one JSON line per move
// ({"state": ..., "time": ms}) from stdin and writes {"move": ...} lines to stdout.
// Used by candidates/brain_js.py; not needed on the server.
"use strict";
const readline = require("readline");
const { performance } = require("perf_hooks");
const brain = require("../brain");
const rl = readline.createInterface({ input: process.stdin });
rl.on("line", (line) => {
  const arrived = performance.now();
  const req = JSON.parse(line);
  if (req.time) brain.setTimeLimit(req.time);
  const r = brain.chooseMove(req.state, arrived);
  process.stdout.write(JSON.stringify({ move: r.move, depth: r.depth || 0 }) + "\n");
});
