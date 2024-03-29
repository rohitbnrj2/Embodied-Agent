from typing import Any
import time

from cambrian.utils.base_config import MjCambrianBaseConfig
from cambrian.utils.config import MjCambrianConfig, run_hydra


def __getattr__(config: MjCambrianBaseConfig, key: str):
    getattr(config, key)


def __getitem__(config: MjCambrianBaseConfig, key: Any):
    config[key]


def __setattr__(config: MjCambrianBaseConfig, key: str, value: Any):
    setattr(config, key, value)


def __setitem__(config: MjCambrianBaseConfig, key: Any, value: Any):
    config[key] = value


def main(config: MjCambrianConfig):
    point_0 = config.env.animals["point_0"]  # noqa
    eyes_lat_range = point_0.eyes_lat_range  # noqa

    num = 100000
    t0 = time.time()
    for _ in range(num):
        # __getattr__(point_0, "use_action_obs")
        # __getitem__(point_0, "use_action_obs")
        # __setattr__(point_0, "use_action_obs", False) # slow
        # __setitem__(point_0, "use_action_obs", False) # slow

        # __getattr__(eyes_lat_range, "start") # error; good
        # __getitem__(eyes_lat_range, slice(0, 1))
        # __setattr__(eyes_lat_range, "start", 0) # error; good
        # __setitem__(eyes_lat_range, slice(0, 1), [0]) # ultra slow

        pass
    t1 = time.time()

    print(f"Time: {(t1 - t0) * 1e3:.2f} ms")
    print(f"Average: {(t1 - t0) * 1e6 / num:.2f} us")


if __name__ == "__main__":
    run_hydra(main)
