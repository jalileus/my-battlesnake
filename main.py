import typing

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
        "version": "v6",
    }


def start(game_state: typing.Dict):
    print("GAME START")


def end(game_state: typing.Dict):
    print("GAME OVER\n")


def move(game_state: typing.Dict) -> typing.Dict:
    return {"move": choose_move(game_state)}


if __name__ == "__main__":
    run_server({"info": info, "start": start, "move": move, "end": end})
