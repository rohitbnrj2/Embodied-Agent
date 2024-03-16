"""This script will create an environment with many mazes and loop through each one
and show the BEV. Each BEV will be saved as an image."""

from pathlib import Path

import cv2
from tqdm.rich import tqdm

from cambrian.env import MjCambrianEnv
from cambrian.utils import setattrs_temporary
from cambrian.utils.config import MjCambrianConfig

if __name__ == "__main__":
    from cambrian.utils.utils import MjCambrianArgumentParser

    parser = MjCambrianArgumentParser()

    parser.add_argument(
        "--output",
        type=str,
        help="Output folder. Defaults to logs/showcase/maze_database",
        default="logs/showcase/maze_database",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed to use for rendering the maze. Defaults to 0",
        default=0,
    )

    args = parser.parse_args()

    # Create our environment
    config = MjCambrianConfig.load(args.config, overrides=args.overrides)

    # Loop through each maze and render the BEV for each and save to an output folder
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    for maze_name in tqdm(config.env_config.maze_configs, desc="Rendering mazes"):
        config.env_config.maze_configs_store[maze_name].custom[
            "maze_adjust_lookat"
        ] = False

        # To implement this, we'll update the maze_config to be the current maze we
        # want to render
        with setattrs_temporary((config.env_config, dict(maze_configs=[maze_name]))):
            # Make the environment here such that only one maze is added
            env = MjCambrianEnv(config.copy())

            # Reset and render the image
            env.reset(seed=args.seed)
            image = env.render()

            # Save the image
            image_path = output_folder / f"{maze_name}.png"
            cv2.imwrite(str(image_path), image[:, :, ::-1])
