from TownScapeENV import *
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import time


# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)

obs_shape = (9, 9)
env = townScape(obs_shape)
env.reset()
model = PPO("MlpPolicy", env, verbose=1)


TIMESTEPS = 2048 #total timesteps per ep
train_eps = 200
for ep in range(1,train_eps+1):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps = False, tb_log_name = "LOG",progress_bar = False)

model_nam = f"{obs_shape[0]}x{obs_shape[1]}_{train_eps}"
model_dir = f"training/{model_nam}"
model.save(model_dir)
del model
env.close()

# model = PPO.load(f"RealModels/{model_nam}")
# action, _ = model.predict(obs)
