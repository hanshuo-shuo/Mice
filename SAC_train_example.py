import argparse
import yaml
from stable_baselines3 import SAC
from oasis_gym import OasisEnv
from reward import *


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_sac(config):
    """
    Use SAC to train OasisEnv.
    """
    env = OasisEnv(
        world_name=config["world_name"],
        use_predator=config["use_predator"],
        reward_function=eval(config["reward_function"]),
        max_step=config["max_step"],
        time_step=config["time_step"],
        render=False,
        real_time=False,
        action_type=OasisEnv.ActionType.CONTINUOUS,
        frame_stack_k=config["frame_stack_k"],
        prey_max_forward_speed=config["prey_max_forward_speed"],
        prey_max_turning_speed=config["prey_max_turning_speed"],
        predator_prey_forward_speed_ratio=config["predator_prey_forward_speed_ratio"],
        predator_prey_turning_speed_ratio=config["predator_prey_turning_speed_ratio"],
    )

    # Create SAC model
    model = SAC(
        "MlpPolicy",
        env,
        seed=config["seed"],
        verbose=1,
        learning_rate=3e-4,
        tensorboard_log="./logs"
    )

    # Train model
    model.learn(total_timesteps=config["total_timesteps"])

    # Save trained model
    save_path = config["save_path"] + config["model_name"]
    model.save(save_path)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    train_sac(config)
