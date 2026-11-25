// Average depth, ms per move and positions/s at a time limit, on 400 saved positions
// (the same sample as candidates/depth_speed.py). Usage: node js/tools/speed.js 130
"use strict";
const fs = require("fs");
const path = require("path");
const { performance } = require("perf_hooks");
const brain = require("../brain");
const limit = Number(process.argv[2] || 130);
const all = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "..", "pypy_test", "data", "positions.json"), "utf8"));
const states = all.filter((_, i) => i % 28 === 0).slice(0, 400);
brain.setTimeLimit(limit);
brain.setMaxDepth(40);
for (const s of states.slice(0, 50)) brain.searchMove(s); // warm up the JIT
let depth = 0, pos = 0, total = 0, max = 0;
for (const s of states) {
  const t = performance.now();
  const [, d, p] = brain.searchMove(s);
  const ms = performance.now() - t;
  depth += d; pos += p; total += ms; max = Math.max(max, ms);
}
console.log(`js ${limit} ms: avg depth ${(depth / states.length).toFixed(2)}, avg ${(total / states.length).toFixed(1)} ms, ` +
  `max ${max.toFixed(1)} ms, ${Math.round(pos / (total / 1000)).toLocaleString()} pos/s`);
