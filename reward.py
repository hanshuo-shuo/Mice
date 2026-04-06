import numpy as np
from torch import le

def custom_reward(obs):
    def _get_from_nonstack(field):
        if hasattr(obs, field):
            return getattr(obs, field)
        if isinstance(obs, np.ndarray):
            stack_fields = [
                "prey_x",
                "prey_y",
                "prey_direction",
                "predator_visible",
                "predator_x",
                "predator_y",
                "predator_direction",
                "near_wall",
                "near_occlusion",
                "time_prey_seen_predator",
            ]
            obs_fields = [
                "prey_x",
                "prey_y",
                "prey_direction",
                "predator_visible",
                "predator_x",
                "predator_y",
                "predator_direction",
                "near_wall", #geometric info
                "near_occlusion",
                "time_prey_seen_predator",
                "puffed",
                "puff_cooled_down",
                "finished",
                "peeking",
                "prey_goal_distance"
            ]
            nonstack_fields = [f for f in obs_fields if f not in stack_fields]
            nonstack = obs[-len(nonstack_fields):]
            return nonstack[nonstack_fields.index(field)]

    reward = 0.0
    if _get_from_nonstack("puffed") > 0:
        reward = -1.0
    if _get_from_nonstack("prey_goal_distance") < 0.1:
        reward = 1.0
    return reward 
