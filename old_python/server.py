import json
import logging
import os
import platform
import time
import typing

from flask import Flask
from flask import request


def run_server(handlers: typing.Dict):
    app = Flask("Battlesnake")

    @app.get("/")
    def on_info():
        return handlers["info"]()

    @app.post("/start")
    def on_start():
        game_state = request.get_json()
        handlers["start"](game_state)
        return "ok"

    @app.post("/move")
    def on_move():
        game_state = request.get_json()
        return handlers["move"](game_state)

    @app.post("/end")
    def on_end():
        game_state = request.get_json()
        handlers["end"](game_state)
        return "ok"

    @app.get("/bench")
    def on_bench():
        # Debug: time one search on a saved position. Never raises, always returns JSON.
        try:
            import brain

            here = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(here, "bench_state.json")
            if not os.path.isfile(path):  # Windows may have saved it as .json.json
                path += ".json"
            with open(path, encoding="utf-8") as f:
                state = json.load(f)
            start = time.perf_counter()
            result = brain.debug_search(state, with_scores=False)  # real time limit
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {
                "depth": result["depth"],
                "positions": result["positions"],
                "elapsed_ms": round(elapsed_ms, 2),
                "move": result["move"],
                "time_limit_ms": brain.TIME_LIMIT_MS,
                "python_version": platform.python_version(),
            }
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}, 500

    @app.after_request
    def identify_server(response):
        response.headers.set(
            "server", "battlesnake/github/starter-snake-python"
        )
        return response

    host = "0.0.0.0"
    port = int(os.environ.get("PORT", "8000"))

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    print(f"\nRunning Battlesnake at http://{host}:{port}")
    app.run(host=host, port=port)
