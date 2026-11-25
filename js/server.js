"use strict";
// Battlesnake web server for the JS brain. Node.js standard library only (no npm packages).
//   GET  /        snake info (look and version)
//   POST /start   game start (nothing to do)
//   POST /move    returns {"move": "up" | "down" | "left" | "right"}
//   POST /end     prints the average search depth of that game
//   GET  /bench   runs one search on bench_state.json and shows depth and time
// Searches run in worker threads (THREADS, default 3), so several games at once each
// keep their own deadline.
// Port: the PORT environment variable (Render sets it), default 8000.
// Time limit: the TIME_LIMIT_MS environment variable, default 150.

const http = require("http");
const { Worker } = require("worker_threads");
const fs = require("fs");
const path = require("path");
const { performance } = require("perf_hooks");
const brain = require("./brain");

if (process.env.TIME_LIMIT_MS) brain.setTimeLimit(Number(process.env.TIME_LIMIT_MS));

// ---- search threads: each move is searched in a worker thread, so the server can
// read the next request (and start its clock) while another game's move is running.
const THREADS = Number(process.env.THREADS || 3);
const idle = [];
const queue = [];
const waiting = new Map(); // job id -> callback
let nextId = 1;
const stats = new Map(); // game id -> [moves, total depth, total positions]

function startWorker() {
  const w = new Worker(path.join(__dirname, "worker.js"));
  w.on("message", (msg) => {
    const cb = waiting.get(msg.id);
    waiting.delete(msg.id);
    w.job = null;
    idle.push(w);
    pump();
    if (cb) cb(msg);
  });
  w.on("error", (e) => {
    console.error("search thread failed:", e);
    if (w.job) {
      const cb = waiting.get(w.job.id);
      waiting.delete(w.job.id);
      if (cb) cb({ move: brain.fallbackMove(w.job.state), depth: 0, positions: 0 });
    }
    startWorker(); // replace it
  });
  w.job = null;
  idle.push(w);
  pump();
}

function pump() {
  while (idle.length && queue.length) {
    const w = idle.pop();
    w.job = queue.shift();
    w.postMessage(w.job);
  }
}

function searchInThread(state, arrivedEpoch, cb) {
  const id = nextId++;
  waiting.set(id, cb);
  queue.push({ id, state, arrivedEpoch, timeLimit: brain.getTimeLimit() });
  pump();
}

for (let i = 0; i < THREADS; i++) startWorker();

const INFO = {
  apiversion: "1",
  author: "jalileus",
  color: "#f13232",
  head: "Evil",
  tail: "Bolt",
  version: "v10-js",
};

function send(res, status, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(status, { "Content-Type": "application/json", Server: "battlesnake/js" });
  res.end(body);
}

function readJson(req, callback) {
  const chunks = [];
  req.on("data", (c) => chunks.push(c));
  req.on("end", () => {
    let data = null;
    try {
      data = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
    } catch (e) {
      data = null;
    }
    callback(data);
  });
}

const server = http.createServer((req, res) => {
  // The move's time budget starts now: if this request had to wait behind another
  // game's move, the wait counts too, so the answer still comes back in time.
  const arrived = performance.now();
  const arrivedEpoch = performance.timeOrigin + arrived;
  const url = req.url.split("?")[0];
  try {
    if (req.method === "GET" && url === "/") return send(res, 200, INFO);

    if (req.method === "GET" && url === "/bench") {
      const file = ["bench_state.json", "bench_state.json.json"]
        .map((f) => path.join(__dirname, f))
        .find((f) => fs.existsSync(f));
      if (!file) return send(res, 500, { error: "bench_state.json not found" });
      const state = JSON.parse(fs.readFileSync(file, "utf8"));
      return searchInThread(state, arrivedEpoch, (r) => send(res, 200, {
        depth: r.depth,
        positions: r.positions,
        elapsed_ms: Math.round((performance.now() - arrived) * 100) / 100,
        move: r.move,
        time_limit_ms: brain.getTimeLimit(),
        node_version: process.version,
      }));
    }

    if (req.method === "POST" && url === "/start") {
      return readJson(req, () => {
        console.log("GAME START");
        send(res, 200, {});
      });
    }

    if (req.method === "POST" && url === "/move") {
      return readJson(req, (state) => {
        if (!state || !state.board || !state.you) return send(res, 400, { error: "bad move request" });
        searchInThread(state, arrivedEpoch, (r) => {
          send(res, 200, { move: r.move });
          const id = state.game && state.game.id;
          if (id && r.depth) {
            if (!stats.has(id) && stats.size >= 200) stats.delete(stats.keys().next().value);
            const s = stats.get(id) || [0, 0, 0];
            s[0]++; s[1] += r.depth; s[2] += r.positions;
            stats.set(id, s);
          }
        });
      });
    }

    if (req.method === "POST" && url === "/end") {
      return readJson(req, (state) => {
        const id = state && state.game ? state.game.id : null;
        const s = id ? stats.get(id) : null;
        if (id) stats.delete(id);
        console.log("GAME OVER");
        if (s)
          console.log(`brain stats: ${s[0]} moves, average depth ${(s[1] / s[0]).toFixed(2)}, ` +
            `average ${Math.round(s[2] / s[0])} positions searched per move`);
        send(res, 200, {});
      });
    }

    send(res, 404, { error: "not found" });
  } catch (e) {
    console.error(e);
    send(res, 500, { error: String(e) });
  }
});

server.keepAliveTimeout = 65000; // keep connections open between moves (less latency)
const port = Number(process.env.PORT || 8000);
server.listen(port, "0.0.0.0", () => {
  console.log(`Battlesnake (JS) running on port ${port}, time limit ${brain.getTimeLimit()} ms, Node ${process.version}`);
});
