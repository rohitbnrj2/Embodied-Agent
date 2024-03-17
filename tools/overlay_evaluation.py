from pathlib import Path
import glob
from functools import partial

import numpy as np
import mujoco as mj
from stable_baselines3.common.vec_env import DummyVecEnv

from cambrian.renderer.renderer import MjCambrianImageViewerOverlay, MjCambrianCursor
from cambrian.env import MjCambrianEnv
from cambrian.utils.wrappers import make_single_env
from cambrian.ml.model import MjCambrianModel
from cambrian.utils import evaluate_policy
from cambrian.utils.config import MjCambrianConfig


def natural_sort(lst):
    """Sort the given list in the way that humans expect."""
    import re

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(lst, key=alphanum_key)


EVAL = 0
NUM_EVALS = 0


def callback(env: MjCambrianEnv):
    if env.episode_step % 2 != 0:
        return

    COLOR = 100
    COLOR = EVAL / NUM_EVALS * (250 - COLOR) + COLOR
    COLOR = np.uint8(((COLOR / 255) ** 2.2) * 255)  # gamma correction
    COLOR = np.array([COLOR, COLOR, COLOR])
    SIZE = [2, 2]
    # INITIAL_POS = np.array([443, 415]) # with overlays
    # SPEED_FACTOR = 15.65 # with overlays
    INITIAL_POS = np.array([475, 355])  # without overlays
    SPEED_FACTOR = 19.62  # without overlays

    animal = next(iter(env.animals.values()))
    delta = np.flip(animal.pos - animal.init_pos)
    delta[0] *= -1
    pos = (INITIAL_POS - delta * SPEED_FACTOR).astype(int)

    obj = np.full((*SIZE, 3), COLOR, dtype=np.uint8)

    cursor = MjCambrianCursor(*pos)
    overlay = MjCambrianImageViewerOverlay(obj, cursor)
    env.overlays[
        f"Tracked Position {env.episode_step + (env.num_resets * env.max_episode_steps)}"
    ] = overlay


def main(args):
    folder = Path(args.folder)
    output_folder = folder / "output"
    output_folder.mkdir(parents=True, exist_ok=True)
    evals_folder = folder / "evaluations"
    assert evals_folder.exists(), f"{evals_folder} does not exist."

    overrides = convert_overrides_to_dict(args.overrides)
    config = MjCambrianConfig.load(folder / "config.yaml", overrides=overrides)
    config.env_config.add_overlays = False
    config.env_config.renderer_config.camera_config.distance_factor = 1.0
    config.env_config.renderer_config.camera_config.lookat = [0, 0, 0]

    env = DummyVecEnv([make_single_env(config, 0)])
    cambrian_env: MjCambrianEnv = env.envs[0].unwrapped

    def _run_eval(config: MjCambrianConfig, pkl: Path):
        assert Path(config.training_config.checkpoint_path).exists()
        model = MjCambrianModel.load(config.training_config.checkpoint_path)
        model.load_rollout(pkl)

        evaluate_policy(
            env,
            model,
            1,
            step_callback=partial(callback, cambrian_env),
        )
        global EVAL
        EVAL += 1

    cambrian_env.record = True
    global NUM_EVALS
    NUM_EVALS = len(list(glob.glob(f"{evals_folder}/*.pkl")))
    for i, file in enumerate(natural_sort(glob.glob(f"{evals_folder}/*.pkl"))):
        print(f"Running evaluation for {file}")
        _run_eval(config, Path(file))

    record_path = output_folder / "final"
    cambrian_env.renderer.save(record_path, save_types=["gif", "mp4", "png", "webp"])
    cambrian_env.record = False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse the evolution folder.")

    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--dry-run", action="store_true", help="Dry run.")
    parser.add_argument(
        "-o",
        "--override",
        dest="overrides",
        action="append",
        nargs=2,
        help="Override config values. Do <dot separated yaml config> <value>",
        default=[],
    )

    parser.add_argument("folder", type=str, help="The folder to parse.")

    args = parser.parse_args()

    main(args)
