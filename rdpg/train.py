import os
import copy
import random

import gym
import numpy as np
import pickle
import torch

from rdpg import RDPG

seed = 777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# environment
env_id = "gym_subt:subt-cave-forward-v0"
env = gym.make(env_id)
env = env.unwrapped
env.seed(seed)

# parameters
num_frames = 4000000
max_epi_step = 512

env.max_step = max_epi_step

agent = RDPG(
    env,
    memory_size=1000,      # numbers of episodes to save
    epi_batch_size=20,         # numbers of sampled episodes
    max_epi_step=max_epi_step,
    initial_random_epi = 100,
    ou_noise_theta=1.0,
    ou_noise_sigma=0.1,
    gamma=0.99,
    tau=5e-3,
    lambda_l2=1e-4,       # l2 regularization weight
)


agent.train(num_frames)
