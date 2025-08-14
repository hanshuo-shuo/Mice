# SAC_train.py 
from stable_baselines3 import SAC
from env import BotEvadeEnv
from reward import custom_reward



def train_sac():
    """
    使用SAC模型训练BotEvadeEnv环境。
    """
    # 初始化环境
    # 为SAC算法选择连续动作空间
    env = BotEvadeEnv(world_name="21_05", 
                  use_lppos=False, 
                  use_predator=True, 
                  reward_function=custom_reward,
                  max_step=300,
                  time_step=0.25,
                  render=False,
                  real_time=False,
                  action_type=BotEvadeEnv.ActionType.CONTINUOUS) # SAC使用连续动作空间

    # 创建SAC模型
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        tensorboard_log="./logs"
    )
    # 训练模型
    model.learn(total_timesteps=200000)
    # 保存训练好的模型
    model.save("sac")
    # 关闭环境
    env.close()

if __name__ == "__main__":
    train_sac() 