from gymnasium.envs.registration import register

register(
    id='mj-cambrian-v0',
    entry_point='cambrian.evolution_envs.three_d.mujoco.env:MjCambrianEnv',
    max_episode_steps= 700,
)