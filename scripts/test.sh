#!/bin/bash

MUJOCO_GL=${MUJOCO_GL:-egl} python cambrian/ml/trainer.py --test evo=evo $@ -m
