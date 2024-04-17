from typing import Any
import time

from cambrian.utils.config import run_hydra, MjCambrianBaseConfig
from cambrian.utils.config.config import MjCambrianConfig


def __getattr__(config: MjCambrianBaseConfig, key: str):
    getattr(config, key)


def __getitem__(config: MjCambrianBaseConfig, key: Any):
    config[key]


def __setattr__(config: MjCambrianBaseConfig, key: str, value: Any):
    setattr(config, key, value)


def __setitem__(config: MjCambrianBaseConfig, key: Any, value: Any):
    config[key] = value


def main(config: MjCambrianConfig):
    a = {"a": 1, "b": 2}  # noqa
    b = [1, 2, 3, 4, 5]  # noqa
    animal_0 = next(iter(config.env.animals.values()))  # noqa
    eyes_lat_range = animal_0.eyes_lat_range  # noqa

    num = 100000
    t0 = time.time()
    for _ in range(num):
        __getattr__(animal_0, "use_action_obs")
        # __getitem__(animal_0, "use_action_obs")
        # __setattr__(animal_0, "use_action_obs", False)  # slow
        # __setattr__(animal_0, "eyes_lat_range", [0, 1]) # ultra slow
        # __setattr__(animal_0, "eyes_lat_range", "test") # ultra slow
        # __setitem__(animal_0, "use_action_obs", False) # slow

        # __getattr__(eyes_lat_range, "start") # error; good
        # __getitem__(eyes_lat_range, slice(0, 1))
        # __setattr__(eyes_lat_range, "start", 0) # error; good
        # __setitem__(eyes_lat_range, slice(0, 1), [0]) # ultra slow

        pass
    t1 = time.time()

    print(f"Time: {(t1 - t0) * 1e3:.2f} ms")
    print(f"Average: {(t1 - t0) * 1e6 / num:.2f} us")


if __name__ == "__main__":
    run_hydra(main, is_readonly=False)
