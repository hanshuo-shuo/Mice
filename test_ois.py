from oasis_env import OasisEnv
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer

import math
goal_locations = [
    (0.75, 0.25), 
    (0.75, 0.75),
    (0.25, 0.75),
]


def my_reward_fn(obs):
    """
    正确的reward函数
    
    重要区别：
    - goal_achieved: 在goal附近累积时间（持续多步）
    - goal_just_completed: 刚完成一个goal（只有1步！）
    """
    reward = 0.0
    
    # 方案1: Sparse reward - 只奖励完成goal
    if obs.goal_just_completed > 0.5:
        reward += 1.0  # 完成一个goal
    
    if obs.finished > 0.5 and obs.goals_remaining == 0:
        reward += 1.0  # 完成所有任务
    
    if obs.puffed > 0.5:
        reward -= 1.0
    
    
    return reward


def random_action():

    # Create environment
    env = OasisEnv(
        world_name="oasis_island7_02",
        goal_locations=goal_locations,
        use_predator=True,
        max_step=100000,
        reward_function=my_reward_fn, 
        render=True,
        action_type=OasisEnv.ActionType.CONTINUOUS,
        real_time=False
    )

    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

def train_sac():

    """
    使用SAC模型训练OasisEnv环境。
    """
    env = OasisEnv(world_name="oasis_island7_02", 
                  goal_locations=goal_locations,
                  use_predator=False,
                  max_step=300,
                  reward_function=my_reward_fn,  # Your custom reward
                  render=False,
                  action_type=OasisEnv.ActionType.CONTINUOUS,
                  predator_speed_multiplier=0.15,
                  real_time=False)
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        tensorboard_log="./logs"
    )
    model.learn(total_timesteps=50000)
    model.save("oasis_sac_no_predator")
    env.close()


def eval_sac(model_path="oasis_sac", 
             n_episodes=100, 
             render=True):
    """
    评估已经训练好的SAC模型在BotEvadeEnv环境的表现。
    """
    # 注意：评估时通常关闭探索噪声，render打开可视化
    env = OasisEnv(world_name="oasis_island7_02", 
                  goal_locations=goal_locations,
                  use_predator=True,
                  max_step=100,
                  reward_function=my_reward_fn,  # Your custom reward
                  render=True,
                  action_type=OasisEnv.ActionType.CONTINUOUS,
                  real_time=True)
    model = SAC.load(model_path)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # 在评估时, 用deterministic动作
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            episode_reward += reward
            if render:
                try:
                    env.model.view.render()
                except Exception:
                    pass
        print(f"Episode {ep+1} reward: {episode_reward}")
        rewards.append(episode_reward)
    print(f"平均reward: {np.mean(rewards)}")
    env.close()

if __name__ == "__main__":
    train_sac()
    # eval_sac()
    # random_action()