#!/bin/bash

[ $# -eq 0 ] && (echo "Please provide the script" && return 0)
SCRIPT=$1
shift

cmd="MUJOCO_GL=egl bash $SCRIPT hydra/launcher=supercloud $@"
>&2 echo "Running command: $cmd"
eval $cmd
