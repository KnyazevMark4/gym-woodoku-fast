## Welcome to gym-woodoku-fast!

An efficient implementation of a reinforcement learning environment for Woodoku game.

It is reinforcement learning environment for "Woodoku" game. 
The repository is inspired by [gym-woodoku](https://github.com/helpingstar/gym-woodoku).
The aim of the project to increase environment computation speed.  
**Key improvements:**
- All positions (81 options) of each figure are precomputed. So now each particular block is extended to array of size (81, 9, 9). This is more efficient way than python loop.
- Calculation of square filling is made by array reshaping and changing axis order.

Also, the project does not use graphics display (pygame) to keep the calculation time to a minimum.
## Installation

```bash
git clone https://github.com/KnyazevMark4/gym-woodoku-fast.git
cd gym-woodoku-fast
pip install -e .
```

## Usage

```python
import gym_woodoku_fast
import gymnasium as gym

env = gym.make('gym_woodoku_fast/WoodokuFast-v0')

obs, info = env.reset()
for i in range(100000):
    action = env.action_space.sample()
    obs, reward, terminated, _, info = env.step(action)
    if terminated:
        obs, info = env.reset()
env.close()
```

## Observation and Action format
Observation and Action format are the same as in the original repo [gym-woodoku](https://github.com/helpingstar/gym-woodoku). 
Score formula is also the same.

