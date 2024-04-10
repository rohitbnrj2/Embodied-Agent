#!/bin/bash

[ $# -eq 0 ] && (echo "Please provide the script to run" && return 0)
SCRIPT=$1
shift

cmd="MUJOCO_GL=${MUJOCO_GL:-egl} bash $SCRIPT hydra/launcher=local $@"
>&2 echo "Running command: $cmd"
eval $cmd
