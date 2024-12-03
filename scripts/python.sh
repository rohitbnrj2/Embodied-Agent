#!/bin/bash

[ $# -eq 0 ] && (echo "Please provide the Python script to run" && exit 1)

SCRIPT=$1
shift

MUJOCO_GL=${MUJOCO_GL:-egl} python "$SCRIPT" "$@"
