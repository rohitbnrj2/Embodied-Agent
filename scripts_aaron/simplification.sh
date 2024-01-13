#!/bin/bash

for exp in r1 r2 r3 r4 r5; do
    MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py \
        configs_mujoco/experiments/simplification/${exp}.yaml --train $@
done
