"""
Example: Training OasisEnv with SAC + HER (Hindsight Experience Replay)

HER improves sample efficiency by learning from failed trajectories:
- If agent fails to reach goal A, HER relabels trajectory as if goal was wherever agent ended up
- This creates "successful" trajectories from failures, greatly improving generalization
"""

from oasis_env import OasisEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
import numpy as np

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

def train_sac_her():
    """
    Train SAC with HER on goal-conditioned OasisEnv.
    """
    print("Creating goal-conditioned environment...")
    
    # Create goal-conditioned environment
    env = OasisEnv(
        world_name="oasis_island7_02", 
        goal_locations=goal_locations,
        use_predator=False,
        max_step=300,
        render=False,
        action_type=OasisEnv.ActionType.CONTINUOUS,
        predator_speed_multiplier=0.15,
        real_time=False,
        goal_conditioned=True,
        reward_function=my_reward_fn,
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test observation format
    obs, _ = env.reset()
    print(f"\nSample observation:")
    print(f"  observation shape: {obs['observation'].shape}")
    print(f"  achieved_goal: {obs['achieved_goal']}")
    print(f"  desired_goal: {obs['desired_goal']}")
    
    # Create SAC model with HER
    print("\nCreating SAC + HER model...")
    model = SAC(
        "MultiInputPolicy",  # Must use MultiInputPolicy for Dict observation!
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,  # Number of virtual transitions per real transition
            goal_selection_strategy='future',  # Sample future states as goals
        ),
        verbose=1,
        learning_rate=3e-4,
        tensorboard_log="./logs_her",
        learning_starts=1000,
    )
    
    print("\nStarting training...")
    model.learn(total_timesteps=50000, log_interval=10)
    
    print("\nSaving model...")
    model.save("oasis_sac_her_no_predator")
    env.close()
    print("Training complete!")


def eval_sac_her(model_path="oasis_sac_her_no_predator", n_episodes=10, render=True):
    """
    Evaluate SAC+HER model on goal-conditioned OasisEnv.
    Test generalization by using different goal sequences.
    """
    print(f"Loading model from {model_path}...")
    
    # Create environment for evaluation
    env = OasisEnv(
        world_name="oasis_island7_02", 
        goal_locations=goal_locations,
        use_predator=True,
        max_step=300,
        render=render,
        action_type=OasisEnv.ActionType.CONTINUOUS,
        predator_speed_multiplier=0.15,
        real_time=False,
        goal_conditioned=True,
        reward_function=my_reward_fn  # Need to pass reward function
    )
    
    # Must pass env when loading HER models!
    model = SAC.load(model_path, env=env)
    
    success_count = 0
    rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        print(f"\nEpisode {ep+1}/{n_episodes}")
        print(f"  Initial position: {obs['achieved_goal']}")
        print(f"  Target goal: {obs['desired_goal']}")
        
        while not done:
            # Get action from model (deterministic for evaluation)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            if render:
                try:
                    env.model.view.render()
                except Exception:
                    pass
        
        is_success = info.get("is_success", 0)
        success_count += is_success
        rewards.append(episode_reward)
        
        print(f"  Steps: {steps}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Success: {'✓' if is_success else '✗'}")
        print(f"  Goals completed: {info.get('goals_completed', 0)}")
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"  Success rate: {success_count}/{n_episodes} ({100*success_count/n_episodes:.1f}%)")
    print(f"  Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"{'='*50}")
    
    env.close()


def test_goal_conditioned():
    """
    Test goal-conditioned observation format.
    """
    print("Testing goal-conditioned environment...")
    
    env = OasisEnv(
        world_name="oasis_island7_02", 
        goal_locations=goal_locations,
        use_predator=True,
        max_step=100,
        render=False,
        action_type=OasisEnv.ActionType.CONTINUOUS,
        goal_conditioned=True
    )
    
    obs, _ = env.reset()
    
    print("\nObservation structure:")
    print(f"  Keys: {obs.keys()}")
    print(f"  observation: {obs['observation'][:5]}... (shape: {obs['observation'].shape})")
    print(f"  achieved_goal: {obs['achieved_goal']} (shape: {obs['achieved_goal'].shape})")
    print(f"  desired_goal: {obs['desired_goal']} (shape: {obs['desired_goal'].shape})")
    
    # Test compute_reward
    reward = env.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})
    print(f"\nReward: {reward}")
    
    # Take a few random actions
    print("\nTaking 5 random actions...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.3f}, achieved={obs['achieved_goal']}, desired={obs['desired_goal']}")
        if terminated or truncated:
            break
    
    env.close()
    print("\nTest complete!")


if __name__ == "__main__":
    # train_sac_her()
    eval_sac_her()
