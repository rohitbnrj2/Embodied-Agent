# Mujoco stuff

Make sure you clone and install using `pip install -e .`.

## TL;DR

Run a simple environment and save the run as a gif/mp4.

```bash
MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/env.py configs_mujoco/point_1lat1lon_1x1.yaml --record-path test -t 200
```

## Visualizing the world/environment

There is a runner in `env.py` that will visualize the world. You have a few run options:

```bash
# Interactive + Display 
# Run with the custom visualization viewer in birds-eye view mode. This is interactive, so you can move around.
python cambrian/evolution_envs/three_d/mujoco/env.py <CONFIG_PATH> -o env_config.renderer_config.render_modes "[human, rgb_array]"

# Noninteractive + headless
# Run the custom viewer but headless and save the output
# NOTE: This is non-interactive and should probably be run headless
python cambrian/evolution_envs/three_d/mujoco/env.py <CONFIG_PATH> --record-path <OUTPUT> -t <TOTAL_TIMESTEPS>

# Noninteractive + headless
# Run with builtin mujoco viewer
python cambrian/evolution_envs/three_d/mujoco/env.py <CONFIG_PATH> --mj-viewer
```

You can pass `-h` to see all options. For more details on the mujoco viewer, [see below](#mujoco-viewer).

### Mujoco Viewer

> [!NOTE]
> There was a bug in the visualizer in 3.0.0, but was fixed in 3.0.1, so run `pip install mujoco --upgrade` to get the latest package.

Hover over an option on the left side and right click to show all the shortcuts. `Q` will visualize the cameras and their frustums. You can also just click it under **Rendering**.

## Running training

```bash
MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py CONFIG_PATH --train -r 0
```

> [!TIP]
> Training should always be done with `MUJOCO_GL=egl` cause that runs with a headless implementation of OpenGL and is significantly faster. `evo.py` will set this automatically for training runs, but you need to explicitly set it for `trainer.py`.

> [!NOTE]
> You can also pass `--eval` to run evaluation to visualize the env. Set the `render_modes` to include `'human'` or pass `--record` to output a gif/mp4. Use `-h` to see options.

## Running evo

```bash
python cambrian/evolution_envs/three_d/mujoco/evo.py CONFIG_PATH
```

> [!NOTE]
> This will spawn `evo_config.population_config.size` individual `trainer.py` calls, where each `trainer.py` has `evo_config.max_n_envs // evo_config.population_config.size` parallel envs, so make sure you aren't launching more envs than cpus on your computer.

## Running on Supercloud

In order to run on supercloud, you need to set `MUJOCO_GL=egl`, which sets OpenGL to use the EGL backend which is headless.

You also can pass `--record-path` to the `env.py` script to set the path that an `mp4` and a `gif` will be recorded.

## Other things

### Configs/Overrides

All configs should be put under `configs_mujoco`. We will transition to use `omegaconf` soon, but for now, you can either edit the config directly in `configs_mujoco` (probably don't want to commit those changes) or use `-o <dot.separated.path> <value>` as used [above](#visualizing-the-worldenvironment).