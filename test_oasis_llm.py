from oasis_env import OasisEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import json
from datetime import datetime
import os
from tqdm import tqdm

# Define goal locations
goal_locations = [
    (0.75, 0.25), 
    (0.75, 0.75),
    (0.25, 0.75),
    (0.25, 0.25),
    (0.5, 0.5),
    (0.5, 0.25),
    (0.5, 0.75),
    (0.25, 0.5),
    (0.75, 0.5),
    (0.75, 0.25),
    (0.75, 0.75),
    (0.25, 0.25),
    (0.5, 0.5),
]

def my_reward_fn(obs):
    reward = 0.0
    if obs.puffed > 0.5:
        reward -= 1.0
    if obs.goal_just_completed > 0.5:
        reward += 1.0
    if obs.finished > 0.5 and obs.goals_remaining == 0:
        reward += 1.0
    return reward

def eval_llm():
    

if __name__ == "__main__":
    # train_sac_her()
    eval_llm()