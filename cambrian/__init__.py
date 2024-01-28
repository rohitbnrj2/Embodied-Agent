from gymnasium.envs.registration import register

register(
    id='mj-cambrian-v0',
    entry_point='cambrian.env:MjCambrianEnv',
    max_episode_steps= 700,
)