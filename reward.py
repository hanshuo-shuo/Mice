def custom_reward(obs):
    reward = 0.0
    if obs.puffed > 0:
        reward = -1.0
    if obs.prey_goal_distance < 0.1:
        reward = 1.0
    return reward