#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --qos=high
#SBATCH --job-name mj_training
#SBATCH --output=out/R-%x.%j_%a.out
#SBATCH --error=out/R-%x.%j_%a.err

[ $# -eq 0 ] && (echo "Please provide the script" && return 0)
SCRIPT=$1
shift

# We'll specify the launcher as supercloud and increase the number of available workers
# to 4 for the sweeper.
cmd="MUJOCO_GL=egl bash $SCRIPT hydra/launcher=supercloud hydra.sweeper.optim.num_workers=4 $@"
echo "Running command: $cmd" | tee /dev/stderr
eval $cmd
