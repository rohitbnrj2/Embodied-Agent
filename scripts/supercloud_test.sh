#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem-per-cpu=2000
#SBATCH --partition=xeon-p8
#SBATCH --job-name=evo
#SBATCH --output=out/R-%x.%j_%a.out
#SBATCH --error=out/R-%x.%j_%a.err

[ $# -eq 0 ] && (echo "Please provide the optimizer" && return 0)
OPT=$1
shift

cmd="MUJOCO_GL=egl bash scripts/test.sh exp=environmental_effects/detection_numeyes1_res0_lon1_fov0 $@ trainer/fitness_fn=test_num_eyes expname='evo/${OPT}' hydra.sweeper.optim.optimizer=${OPT} hydra/launcher=joblib hydra.sweeper.optim.load_if_exists=null -m"
echo "Running command: $cmd"
eval $cmd

cmd="MUJOCO_GL=egl python tools/parse_evos/parse_evos.py folder=logs/$(date +%F)/evo/${OPT} plot=True force=True config_filename=test_config.yaml plots_mask='[fov0_vs_generation,lon0_vs_generation,resolution_vs_generation,number_of_eyes_vs_generation,fitness_vs_generation]' check_finished=False"
echo "Running command: $cmd"
eval $cmd
