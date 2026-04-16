import yaml
import os
from datetime import datetime

model_name = f"sac_peeking_{datetime.now().strftime('%m%d')}"



config = {
    'model_name': model_name,
    'seed': 3456, # None for random seed
    "total_timesteps": 100000,
    "time_step": 0.1,
    "world_name": "clump01_05", # "21_05" for scatter world
    "use_lppos": False,
    "use_predator": True,
    "reward_function": "custom_reward",
    "max_step": 300,
    "predator_prey_forward_speed_ratio": 0.3, 
    'frame_stack_k': 3,
    'save_path': "./Saved_Models/"
}

# Save config with timestamp
exp_dir = "./configs"

with open(os.path.join(exp_dir, f"{model_name}.yaml"), 'w') as f:
    yaml.dump(config, f)