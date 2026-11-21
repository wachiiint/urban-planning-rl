"""
Raw environment smoke test
Simulate random action from env, then the result is shown.
Run from the repo root:  python src/test.py
"""
from icecream import ic
from TownScapeENV import *

obs_shape = (10,10)
env = townScape(obs_shape)
ic(env.observation_shape)
env.reset()
ic(env.obs)

done = False
while not done:
    act = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(act)
    done = terminated or truncated
ic('dones')
Image.fromarray(env.render()).show()
