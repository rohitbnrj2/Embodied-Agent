# EyesOfCambrian

Install the `cambrian` package by doing:
```
pip install -e .
```

## TL;DR

Run the following to create a simple environment and save the run as a video. 
The `--record` flag optionally takes an integer argument to specify the number of 
timesteps to record. The resulting video will be saved to 
`logs/<the date>/example/eval.*`.

```bash
python cambrian/envs/env.py run_renderer exp=example --record 100
```

## Visualizing the world/environment

There is a runner in `env.py` that will visualize the world. You have a few run options:

```bash
# Interactive + Display
# Run with the custom visualization viewer in birds-eye view mode. This is interactive, 
# so you can pan the renderer around using the mouse.
python cambrian/envs/env.py run_renderer exp=<EXPERIMENT> env.renderer.render_modes="[human]"

# Noninteractive + headless
# Run the custom viewer but headless and save the output
# NOTE: This is non-interactive and should probably be run headless
python cambrian/envs/env.py run_renderer exp=<EXPERIMENT> --record <OPTIONAL_TOTAL_TIMESTEPS>

# Interactive + Display
# Run with built in mujoco viewer. You will need to scroll out to see the full view.
python cambrian/envs/env.py run_mj_viewer exp=<EXPERIMENT>
```

You can pass `-h` to see all options. Look in `configs/exp` for all the experiments 
you can run. For more details on the mujoco viewer, [see below](#mujoco-viewer).

### Custom Viewer

This is a custom viewer that we use for debugging. There are a few shortcuts you can 
use:

- `R`: Reset the environment
- `Tab`: Switch cameras in the main view
- `Space`: Pause the simulation
- `Exit`: Close the window

### Mujoco Viewer

Hover over an option on the left side and right click to show all the shortcuts. `Q` will visualize the cameras and their frustums. You can also just click it under **Rendering**.

## Running training

```bash
bash scripts/local.sh scripts/train.sh exp=<EXPERIMENT>
```

> [!TIP]
> The `local.sh` script will automatically set `MUJOCO_GL=egl`. Training should always 
be done with `MUJOCO_GL=egl` cause that runs with a headless implementation of OpenGL 
and is significantly faster. 

> [!NOTE]
> You can also run `eval.sh` to run evaluation to visualize the env. 
Set the `render_modes` to include `'human'` to visualize the env. It will also 
automatically save the video to `logs/<date>/<exp>/eval.*`. Use `-h` to see options.

## Running evo

```bash
bash scripts/local.sh scripts/train.sh exp=<EXPERIMENT> -m
```

Here, `-m` stands for `--multirun`. This will spawn multiple sweep instances of the
`trainer.py` script. See the config files sweep parameters and/or 
[hydra](https://hydra.cc/docs/intro) for more details.

> [!NOTE]
> You may get an error saying something along the lines 
> `No variable to optimize in this parametrization.` This is because the config you 
> chose doesn't have sweep parameters. You can either add some via the command line 
> (see the [hydra documentation](https://hydra.cc/docs/intro)), add them to the
> config file or choose another exp.

## Running on Supercloud

To run on supercloud, replace `local.sh` with `sc.sh` in the above commands. This will
set various environment variables that are required to run on supercloud. You can 
still run `local.sh` if you don't plan to use slurm to submit a job.

> [!NOTE]
> See `configs/hydra/launcher/supercloud.yaml` and 
> `configs/hydra/sweeper/evolution.yaml` for the slurm and evolution settings, 
> respectively.

### Training

```bash
# If running headless
sbatch scripts/sc.sh scripts/train.sh exp=<EXPERIMENT>
# If running in interactive mode
bash scripts/local.sh scripts/train.sh exp=<EXPERIMENT>
```

### Evaluation

```bash
# If running headless
sbatch scripts/sc.sh scripts/eval.sh exp=<EXPERIMENT>
# If running in interactive mode
bash scripts/local.sh scripts/eval.sh exp=<EXPERIMENT>
```

### Evo

```bash
# If running headless
sbatch scripts/sc.sh scripts/train.sh exp=<EXPERIMENT> -m
# If running in interactive mode
bash scripts/local.sh scripts/train.sh exp=<EXPERIMENT> -m
```

## Other things

### Configs/Overrides

All configs should be put under `configs`. These are parsed by 
[hydra](https://hydra.cc/docs/intro) and can be overridden by passing in 
`<dot.separated.path>=<value>` to the script. Checkout hydra's documentation for more 
details.