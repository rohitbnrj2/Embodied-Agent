# Mujoco stuff

## Visualizing the world/environment

There is a runner in `env.py` that will visualize the world. To run, run the following:

```bash
python cambrian/evolution_envs/three_d/mujoco/env.py CONFIG_PATH
```

By default, it will use the builtin BEV viewer I made for the visualizations. If you want to visualize cameras or do more complicated stuff, pass `--mj-viewer`. This uses the Mujoco simulation viewer. See below for more details. You can pass `-h` for all options.

### Mujoco Viewer

> [!NOTE]
> There was a bug in the visualizer in 3.0.0, but was fixed in 3.0.1, so run `pip install mujoco --upgrade` to get the latest package.

Hover over an option on the left side and right click to show all the shortcuts. `Q` will visualize the cameras and their frustums. You can also just click it under **Rendering**.

## Running training

```bash
python cambrian/evolution_envs/three_d/mujoco/runner.py CONFIG_PATH --evo -r 0
```

> [!TIP]
> Training should always be done with `MUJOCO_GL=egl` cause that runs in headless mode and is significantly faster.

## Running on Supercloud

In order to run on supercloud, you need to set `MUJOCO_GL=egl`, which sets OpenGL to use the EGL backend which is headless.

You also can pass `--record-path` to the `env.py` script to set the path that an `mp4` and a `gif` will be recorded.