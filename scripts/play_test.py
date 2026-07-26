import random
import sys
import os

from cg.api import to_observation_class
from cg.game import battle_start, battle_select, battle_finish


_SAMPLE_DECK = [
    673, 673, 674, 674, 675, 675, 676, 676,
    676, 677, 677, 677, 678, 678, 678, 678,
    1102, 1102, 1102, 1102, 1123, 1123, 1141, 1141,
    1141, 1141, 1142, 1142, 1142, 1142, 1152, 1152,
    6, 1159, 1182, 1182, 1192, 1192, 1192, 1192,
    1227, 1227, 1227, 1227, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6,
    6, 1182, 677, 1252,
]


def random_agent(obs_dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return _SAMPLE_DECK
    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)


def _play_one(agent0_fn, deck0, agent1_fn, deck1, max_steps=1000):
    obs = None
    try:
        obs, start_data = battle_start(deck0, deck1)
        print("starting battle...")
        print("deck: ", len(deck0))
        print("deck: ", len(deck1))
        if obs is None:
            return None, 0, f"battle_start_failed:p{start_data.errorPlayer}:{start_data.errorType}"
        for _step in range(max_steps + 1):
            obc = to_observation_class(obs)
            if obc.current.result >= 0:
                return obc.current.result, _step, ""
            active_fn = agent0_fn if obc.current.yourIndex == 0 else agent1_fn
            obs = battle_select(active_fn(obs))
        return None, max_steps, "max_steps"
    except Exception as exc:
        return None, 0, f"{type(exc).__name__}:{exc}"
    finally:
        if obs is not None:
            print("done!")
            battle_finish()



def main():
    _play_one(random_agent, _SAMPLE_DECK, random_agent, _SAMPLE_DECK)


if __name__ == "__main__":
    main()
