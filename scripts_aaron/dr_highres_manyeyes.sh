experiment=point_16eyes_10x10
include='${extend:${include:configs_mujoco/experiments/'$experiment'.yaml}}'

MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py \
    configs_mujoco/experiments/dr.yaml --train \
    -o extend.experiment=$include \
    -o training_config.total_timesteps=5_000_000 -ao n_temporal_obs=1 \
    -o training_config.exp_name='dr_n_temporal_obs_1'

MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py \
    configs_mujoco/experiments/dr.yaml --train \
    -o extend.experiment=$include \
    -o training_config.total_timesteps=5_000_000 -ao n_temporal_obs=5 \
    -o training_config.exp_name='dr_n_temporal_obs_5'

MUJOCO_GL=egl python cambrian/evolution_envs/three_d/mujoco/trainer.py \
    configs_mujoco/experiments/dr.yaml --train \
    -o extend.experiment=$include \
    -o training_config.total_timesteps=5_000_000 -ao n_temporal_obs=10 \
    -o training_config.exp_name='dr_n_temporal_obs_10'
