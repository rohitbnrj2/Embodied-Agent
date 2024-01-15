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

source /etc/profile
module load cuda/11.8

module load anaconda/2023b
source "/state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/etc/profile.d/conda.sh"
/state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/condabin/conda activate /home/gridsan/ayoung/Pseudos/bees/.conda/envs/bees 

export TF_CPP_MIN_LOG_LEVEL=2
export OPENBLAS_NUM_THREADS=1
export PMIX_MCA_gds=hash
ulimit -u unlimited

REPO=/home/gridsan/ayoung/Pseudos/bees/EyesOfCambrian

[ $# -eq 0 ] && (echo "Please provide the script" && return 0)
SCRIPT=$1

shift

cd $REPO
>&2 echo "Running script..."
MUJOCO_GL=egl bash $SCRIPT -o extend.sc='${extend:${include:configs_mujoco/overlays/supercloud.yaml}}'
