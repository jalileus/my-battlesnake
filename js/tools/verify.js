// Check the JS port against brain.py: same move AND same number of positions searched
// at a fixed depth, on the saved positions. Run tools/dump_expected.py first.
// Usage (from the repo root): node js/tools/verify.js
"use strict";
const fs = require("fs");
const path = require("path");
const brain = require("../brain");

const JS = path.join(__dirname, "..");
const states = JSON.parse(fs.readFileSync(path.join(JS, "..", "pypy_test", "data", "positions.json"), "utf8"));
const expected = JSON.parse(fs.readFileSync(path.join(JS, "test", "expected.json"), "utf8"));
brain.setTimeLimit(1e12);
const result = {};
const t0 = Date.now();
for (const [i, depth, move, positions] of expected) {
  brain.setMaxDepth(depth);
  const [m, , p] = brain.searchMove(states[i]);
  const r = (result[depth] = result[depth] || { n: 0, move: 0, pos: 0, bad: [] });
  r.n++;
  if (m === move) r.move++;
  if (p === positions) r.pos++;
  if ((m !== move || p !== positions) && r.bad.length < 5) r.bad.push([i, move, m, positions, p]);
}
for (const [d, r] of Object.entries(result))
  console.log(`depth ${d}: same move ${r.move}/${r.n}, same positions searched ${r.pos}/${r.n}` +
    (r.bad.length ? `, first differences [index, py move, js move, py pos, js pos]: ${JSON.stringify(r.bad)}` : ""));
console.log(`(${((Date.now() - t0) / 1000).toFixed(0)} s)`);
