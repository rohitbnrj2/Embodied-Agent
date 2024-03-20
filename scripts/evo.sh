#!/bin/bash

MUJOCO_GL=egl python cambrian/ml/evo.py $@ \
    evo.num_nodes=${SLURM_ARRAY_TASK_COUNT:-1} \
    evo.generation.rank=${SLURM_ARRAY_TASK_ID:-0} \
    evo.generation.generation=0 \
