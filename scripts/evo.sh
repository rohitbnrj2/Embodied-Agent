#!/bin/bash

MUJOCO_GL=egl python cambrian/ml/evo.py -r ${SLURM_ARRAY_TASK_ID:-0} $@