#!/bin/bash

MUJOCO_GL=egl python cambrian/ml/evo.py $@ \
    -o evo_config.num_nodes=${SLURM_ARRAY_TASK_COUNT:-1} \
    -o evo_config.generation_config.rank=${SLURM_ARRAY_TASK_ID:-0} \
    -o evo_config.generation_config.generation=0 \
