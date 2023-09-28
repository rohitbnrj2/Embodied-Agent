#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task 40
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:volta:1
#SBATCH --time=4-04:00:00
#SBATCH --qos=high
#SBATCH --job-name mpi_training
#SBATCH --output=out/R-%x.%j_%a.out
#SBATCH --error=out/R-%x.%j_%a.err

source /etc/profile
module load anaconda
module load cuda/11.8

# Fix for > 25 threads issue
export OPENBLAS_NUM_THREADS=1
export PMIX_MCA_gds=hash

REPO=/home/gridsan/ayoung/robobees/EyesOfCambrian-ppo
export CAMBRIAN_IP_DIR=$REPO/ips/

# Hacky: We're going to write our IP to a file that is available to all other agent pools on different nodes/processes
mkdir -p $REPO/ips
echo "$SLURM_ARRAY_TASK_ID $(hostname -i)" > $REPO/logs-distributed-evo/debug/agent_pool_ips/ip_$SLURM_ARRAY_TASK_ID.txt

cd $REPO
source env/bin/activate
python -u cambrian/reinforce/evo/runner.py configs_evo/debug_distributed.yaml
