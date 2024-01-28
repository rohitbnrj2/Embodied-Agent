#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:volta:1
#SBATCH --time=4-04:00:00
#SBATCH --qos=high
#SBATCH --job-name mj_training
#SBATCH --output=out/R-%x.%j_%a.out
#SBATCH --error=out/R-%x.%j_%a.err

source /etc/profile

[ $# -eq 0 ] && (echo "Please provide the script" && return 0)
SCRIPT=$1

export TF_CPP_MIN_LOG_LEVEL=2
export OPENBLAS_NUM_THREADS=1
export PMIX_MCA_gds=hash

shift

>&2 echo "Running script $PWD..."
MUJOCO_GL=egl bash $SCRIPT "$@" 