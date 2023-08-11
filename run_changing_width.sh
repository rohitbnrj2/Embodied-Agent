#!/bin/bash
source /etc/profile

/home/gridsan/ktiwary/.conda/envs/flatland-rl/bin/pip install -e . && python3 cambrian/reinforce/env_v1.py configs/ten_compound_eyes_v1_maze2.yaml
