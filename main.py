import typing

import brain
from brain import choose_move
from server import run_server


def info() -> typing.Dict:
    print("INFO")
    return {
        "apiversion": "1",
        "author": "jalileus",
        "color": "#f13232",
        "head": "Evil",
        "tail": "Bolt",
        "version": "v7",
    }


def start(game_state: typing.Dict):
    print("GAME START")


def end(game_state: typing.Dict):
    print("GAME OVER\n")
    try:
        brain.print_game_stats(game_state["game"]["id"])
    except Exception as e:  # debug output must never crash the server
        print(f"could not print brain stats: {e!r}")


def move(game_state: typing.Dict) -> typing.Dict:
    return {"move": choose_move(game_state)}


if __name__ == "__main__":
    run_server({"info": info, "start": start, "move": move, "end": end})
