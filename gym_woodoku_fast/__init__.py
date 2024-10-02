from gymnasium.envs.registration import register

__version__ = "0.0.1"

register(
    id='gym_woodoku_fast/WoodokuFast-v0',
    entry_point='gym_woodoku_fast.envs:WoodokuFastEnv',
)
