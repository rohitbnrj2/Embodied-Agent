#!/bin/bash
#SBATCH -c 5
#SBATCH --gres=gpu:volta:1

### Tasks for Bhavya: 
# 1. train each of these separately
# 2. Copy the policy weights like we talked about. 
# 3. Copy the policy weights and encoder weights. 
# 4. Create monitor graphs to see how the performance changes.
##################

# parent case 1
MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py configs_mujoco/point_1lat1lon_1x1.yaml --train \
    -o training_config.exp_name=parent_1_point_1lat1lon_1x1 \
    -o training_config.n_envs=5 -o includes.maze=configs_mujoco/mazes/train/G_MAZE_DIVERSE.yaml \
    -o training_config.max_episode_steps=256 -o training_config.total_timesteps=2000000 \
    -o animal_config.n_temporal_obs=5 -o env_config.truncate_on_contact=False

# parent case 2
MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py configs_mujoco/point_1lat1lon_2x2.yaml --train \
    -o training_config.exp_name=parent_2_point_1lat1lon_2x2 \
    -o training_config.n_envs=5 -o includes.maze=configs_mujoco/mazes/train/G_MAZE_DIVERSE.yaml \
    -o training_config.max_episode_steps=256 -o training_config.total_timesteps=2000000 \
    -o animal_config.n_temporal_obs=5 -o env_config.truncate_on_contact=False

# child case from parent 1 this is a a big change, from parent 2 this is a smaller change. 
MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py configs_mujoco/point_1lat2lon_2x2.yaml --train \
    -o training_config.exp_name=child_1_point_1lat2lon_2x2 \
    -o training_config.n_envs=5 -o includes.maze=configs_mujoco/mazes/train/G_MAZE_DIVERSE.yaml \
    -o training_config.max_episode_steps=256 -o training_config.total_timesteps=2000000 \
    -o animal_config.n_temporal_obs=5 -o env_config.truncate_on_contact=False

# child case very different to see how the dynamics model handles larger mutations in eyes. 
# since the dynamics model from parent 1 and 2 are not going to be that good, so you can test with child 1 and see how it does.
MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py configs_mujoco/point_2lat5lon_1x1.yaml --train \
    -o training_config.exp_name=child_1_point_2lat5lon_1x1 \
    -o training_config.n_envs=5 -o includes.maze=configs_mujoco/mazes/train/G_MAZE_DIVERSE.yaml \
    -o training_config.max_episode_steps=256 -o training_config.total_timesteps=2000000 \
    -o animal_config.n_temporal_obs=2 -o env_config.truncate_on_contact=False
