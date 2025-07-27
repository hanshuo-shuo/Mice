# DQN_train.py 
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from env import BotEvadeEnv
from reward import custom_reward



def train_dqn(env):
    env = BotEvadeEnv(world_name="clump01_05", 
                  use_lppos=False, 
                  use_predator=True, 
                  reward_function=custom_reward,
                  max_step=300,
                  time_step=0.25,
                  render=False,
                  real_time=False,
                  action_type=BotEvadeEnv.ActionType.DISCRETE)
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        train_freq=(1, "step"),
        tensorboard_log="./logs"
    )
    
    model.learn(total_timesteps=100000)
    model.save("dqn")
    env.close()

if __name__ == "__main__":
    train_dqn()