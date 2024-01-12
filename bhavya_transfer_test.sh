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
# MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py \
#     configs_mujoco/experiments/point_1eye_1x1.yaml --train \
#     -o training_config.exp_name='parent_1_${..filename}' \
#     -o training_config.max_episode_steps=256 -o training_config.total_timesteps=2000000 \
#     -ao n_temporal_obs=5 -o env_config.truncate_on_contact=False

# # parent case 2
# MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py \
#     configs_mujoco/experiments/point_1eye_2x2.yaml --train \
#     -o training_config.exp_name='parent_2_${..filename}' \
#     -o training_config.max_episode_steps=256 -o training_config.total_timesteps=2000000 \
#     -ao n_temporal_obs=5 -o env_config.truncate_on_contact=False

# # child case from parent 1 this is a a big change, from parent 2 this is a smaller change. 
# MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py \
#     configs_mujoco/experiments/point_2eyes_2x2.yaml --train \
#     -o training_config.exp_name='child_1_${..filename}' \
#     -o training_config.max_episode_steps=256 -o training_config.total_timesteps=2000000 \
#     -ao n_temporal_obs=5 -o env_config.truncate_on_contact=False

# child case very different to see how the dynamics model handles larger mutations in eyes. 
# since the dynamics model from parent 1 and 2 are not going to be that good, so you can test with child 1 and see how it does.
# MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py \
#     configs_mujoco/experiments/point_3eyes_20x1.yaml --train \
#     -o training_config.exp_name='flatland_apopen1_torch_apresEQres_eyes_20x1' \
#     -o training_config.total_timesteps=1000000 \
#     -ao n_temporal_obs=5 -o env_config.truncate_on_contact=False

# MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py \
#     configs_mujoco/experiments/point_2eyes_20x1.yaml --train \
#     -o training_config.exp_name='gamm2_ambient_tunnel_with_optics_apdot2_2eyes' \
#     -o training_config.total_timesteps=1000000 \
#     -ao n_temporal_obs=5 -o env_config.truncate_on_contact=True

# MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py \
#     configs_mujoco/experiments/point_2eyes_20x1.yaml --train \
#     -o training_config.exp_name='gamm2_ambient_tunnel_with_optics_ap1_2eyes' \
#     -o training_config.total_timesteps=2000000 \
#     -ao n_temporal_obs=5 -o env_config.truncate_on_contact=True


# write a bash script that iterates over aperture = 0.2 0.5 and 1
# for aperture in 0.2 0.5 1; do
#     MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py \
#         configs_mujoco/experiments/point_2eyes_20x1.yaml --train \
#         -o training_config.exp_name='gamm4_ambient_tunnel_with_optics_ap${aperture}_2eyes' \
#         -o training_config.total_timesteps=2000000 \
#         -ao n_temporal_obs=5 -o env_config.truncate_on_contact=True \
#         -o env_config.terminate_at_goal=True \
#         -ao eye_configs.animal_0_eye_0.aperture_open=${aperture} \
#         -ao eye_configs.animal_0_eye_1.aperture_open=${aperture}
# done

# MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py \
#     configs_mujoco/experiments/point_2eyes_20x1.yaml --train \
#     -o training_config.exp_name='gamm4_ambient_S_MAZE_STATIC_optics_apdot5_apdot2_2eyes' \
#     -o training_config.total_timesteps=2000000 \
#     -ao n_temporal_obs=5 -o env_config.truncate_on_contact=True

# MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py \
#     configs_mujoco/experiments/point_2eyes_20x1.yaml --train \
#     -o training_config.exp_name='NO_end_at_goal_gamm4_ambient_S_MAZE_STATIC_optics_apdot5_apdot2_2eyes' \
#     -o training_config.total_timesteps=2000000 \
#     -ao n_temporal_obs=5 -o env_config.truncate_on_contact=True
#     # -o env_config.terminate_at_goal=True 

MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py \
    configs_mujoco/experiments/point_2eyes_20x1.yaml --train \
    -o training_config.exp_name='gamm2_ambient_oflowTunnel_optics_apdot5_apdot2_2eyes' \
    -o training_config.total_timesteps=2000000 \
    -ao n_temporal_obs=5 
    # -o env_config.truncate_on_contact=True
    # -o env_config.terminate_at_goal=True 