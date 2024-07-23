"""This script will create an environment and take many images which aid in
visualizing what the agent sees."""

from pathlib import Path

import cv2
import numpy as np
import mujoco as mj

from cambrian.envs import MjCambrianEnv
from cambrian.renderer import resize_with_aspect_fill
from cambrian.utils import setattrs_temporary
from cambrian.utils.config import MjCambrianConfig

if __name__ == "__main__":
    from cambrian.utils.utils import MjCambrianArgumentParser

    parser = MjCambrianArgumentParser()

    parser.add_argument(
        "--output",
        type=str,
        help="Output folder. Defaults to logs/showcase/eye_visualization",
        default="logs/showcase/eye_visualization",
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

    # Create our environment
    env = MjCambrianEnv(config)

    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    def run(filename, num_steps=0):
        # Reset and render the image
        env.reset(seed=args.seed)
        for _ in range(num_steps):
            mj.mj_step(env.model, env.data)
            env.render()
        image = env.render()

        # Save the image
        cv2.imwrite(str(output_folder / f"{filename}.png"), image[:, :, ::-1])

    # First, visualize the BEV (the default)
    temp_attrs = []
    temp_attrs.append(
        (
            env.env_config.maze_configs_store[config.env_config.maze_configs[0]],
            dict(custom={"maze_adjust_lookat": False}),
        )
    )
    with setattrs_temporary(*temp_attrs):
        run(f"{config.filename}_bev")
    temp_attrs.append(
        (env.renderer.viewer.config.scene_options, dict(flags=dict(mjVIS_CAMERA=False)))
    )
    with setattrs_temporary(*temp_attrs):
        run(f"{config.filename}_bev_wo_frustrum")

    # Next, visualize a third person view
    temp_attrs = []
    temp_attrs.append(
        (
            env.renderer.viewer.config.camera_config,
            dict(
                azimuth=-60,
                elevation=-45,
                distance_factor=0.5,
                trackbodyid=1,
                typename="tracking",
            ),
        )
    )
    with setattrs_temporary(*temp_attrs):
        run(f"{config.filename}_third_person", 10)

    # Now, construct a first person composite view
    # Assumes uniform eyes and one animal
    lats, lons = set(), set()
    images = {}
    for animal in env.animals.values():
        max_res = (
            max([eye.config.resolution[0] for eye in animal.eyes.values()]),
            max([eye.config.resolution[1] for eye in animal.eyes.values()]),
        )
        for i, eye in enumerate(animal.eyes.values()):
            lat, lon = eye.config.coord
            if lat not in images:
                images[lat] = {lon: eye.prev_obs}
            else:
                assert lon not in images[lat]
                images[lat][lon] = eye.prev_obs

        # Now sort the images
        lats = sorted(images.keys())
        lons = sorted(images[lats[0]].keys())

        # Now construct the composite image
        composite = []
        for lat in lats:
            composite.append([])
            for lon in lons:
                composite[-1].append(
                    resize_with_aspect_fill(images[lat][lon], *max_res)
                )
        composite = np.vstack([np.hstack(row) for row in composite])
        composite = resize_with_aspect_fill(np.flipud(composite), 1000, 1000) * 255.0

        # Save the image
        cv2.imwrite(
            str(output_folder / f"{config.filename}_{animal.name}_first_person.png"),
            composite[:, :, ::-1],
        )
