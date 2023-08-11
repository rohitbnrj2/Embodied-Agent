#!/bin/bash
source /etc/profile

/home/gridsan/ktiwary/.conda/envs/flatland-rl/bin/pip install -e . && python3 cambrian/reinforce/env_v1.py configs/2compound_eyes_changin_width.yaml
