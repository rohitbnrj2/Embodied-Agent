#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task 40
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:volta:1
#SBATCH --time=4-04:00:00
#SBATCH --qos=high
#SBATCH --job-name mj_training
#SBATCH --output=out/R-%x.%j_%a.out
#SBATCH --error=out/R-%x.%j_%a.err

# source /etc/profile
# module load anaconda/Python-ML-2023b
# module load cuda/11.8

export TF_CPP_MIN_LOG_LEVEL=2
export OPENBLAS_NUM_THREADS=1
export PMIX_MCA_gds=hash
# mj-ktiwary-run.sh: line 19: ulimit: max user processes: cannot modify limit: Operation not permitted
# ulimit -u unlimited 

REPO=/home/gridsan/ktiwary/src/cambrian-main

[ $# -eq 0 ] && (echo "Please provide the config path" && return 0)
CONFIG=$1

shift

cd $REPO
MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/evo.py $CONFIG -o evo_config.max_n_envs=40 -r $SLURM_ARRAY_TASK_ID $@
