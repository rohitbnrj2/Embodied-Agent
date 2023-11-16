# Mujoco stuff

## Visualizing the world/environment

There is a runner in `env.py` that will visualize the world. To run, run the following:

```bash
python cambrian/evolution_envs/three_d/mujoco/env.py CONFIG_PATH
```

By default, it will use the builtin BEV viewer I made for the visualizations. If you want to visualize cameras or do more complicated stuff, pass `--mj-viewer`. You can pass `-h` for all options.

### Mujoco Viewer

Hover over an option on the left side and right click to show all the shortcuts. `Q` will visualize the cameras and their frustums. You can also just click it under **Rendering**.

## Running training

Will update here, currently broken.

## Running on Supercloud

Only difference for running on supercloud is you need to set `MUJOCO_GL=egl`, which sets OpenGL to use the EGL backend which is headless.