#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --qos=sbel
#SBATCH --partition=sbel
#SBATCH --mem-per-cpu=3000
#SBATCH --job-name cambrian
#SBATCH --output=out/R-%x.%j_%a.out
#SBATCH --error=out/R-%x.%j_%a.err

# This is set globally for some reason, and complains when you set slurm mem 
# per gpu at a job level
unset SLURM_MEM_PER_CPU

[ $# -eq 0 ] && (echo "Please provide the script" && return 0)
SCRIPT=$1
shift

# We'll specify the launcher as euler and increase the number of available workers
# to 4 for the sweeper.
cmd="MUJOCO_GL=egl bash $SCRIPT hydra/launcher=euler $@"
echo "Running command: $cmd" | tee /dev/stderr
eval $cmd
