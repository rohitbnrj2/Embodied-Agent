#!/bin/bash
source /etc/profile

/home/gridsan/ktiwary/.conda/envs/flatland-rl/bin/pip install -e . && \
    python3 cambrian/reinforce/env_v2.py ./configs_v3/10_compound_eyes_1024_nsteps_curvy_v2.yaml
