#!/bin/bash

MUJOCO_GL=egl python cambrian/ml/trainer.py --train $@
