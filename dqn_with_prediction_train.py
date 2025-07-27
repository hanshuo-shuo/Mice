# DQN with Prediction Training Script
from dqn_with_prediction import DQNWithPrediction
from env import BotEvadeEnv
from reward import custom_reward
import numpy as np


def train_dqn_with_prediction():
    """Train DQN with state prediction capability"""
    
    env = BotEvadeEnv(
        world_name="clump01_05", 
        use_lppos=False, 
        use_predator=True, 
        reward_function=custom_reward,
        max_step=300,
        time_step=0.25,
        render=False,
        real_time=False,
        action_type=BotEvadeEnv.ActionType.DISCRETE
    )
    
    # Create DQN with prediction model
    model = DQNWithPrediction(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        train_freq=(1, "step"),
        tensorboard_log="./logs_with_prediction",
        # Prediction-specific parameters
        prediction_learning_rate=1e-4,
        prediction_loss_weight=1.0,
        predictor_hidden_dim=256
    )
    
    # Train the model
    print("Training DQN with prediction...")
    model.learn(total_timesteps=400000, log_interval=10)
    
    # Save the model
    model.save("dqn_with_prediction")
    print("Model saved as 'dqn_with_prediction'")
    
    env.close()
    return model


def evaluate_prediction_errors(model_path: str = "dqn_with_prediction"):
    """Evaluate the trained model and analyze prediction errors"""
    
    env = BotEvadeEnv(
        world_name="clump01_05", 
        use_lppos=False, 
        use_predator=True, 
        reward_function=custom_reward,
        max_step=300,
        time_step=0.25,
        render=False,
        real_time=False,
        action_type=BotEvadeEnv.ActionType.DISCRETE
    )
    
    # Load the trained model
    model = DQNWithPrediction.load(model_path, env=env)
    
    # Set evaluation mode to track prediction errors
    model.set_eval_mode(True)
    
    print("Evaluating prediction errors...")
    
    # Run evaluation episodes
    num_episodes = 10
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < 300:
            # Get action from the model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take action and get next observation
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Evaluate prediction error
            prediction_error = model.evaluate_prediction_error(obs, action, next_obs)
            
            obs = next_obs
            step_count += 1
            
            if done or truncated:
                break
        
        print(f"Episode {episode + 1} completed with {step_count} steps")
    
    # Get prediction error statistics
    prediction_errors = model.get_prediction_errors()
    
    if prediction_errors:
        print(f"\nPrediction Error Statistics:")
        print(f"Mean error: {np.mean(prediction_errors):.6f}")
        print(f"Std error: {np.std(prediction_errors):.6f}")
        print(f"Min error: {np.min(prediction_errors):.6f}")
        print(f"Max error: {np.max(prediction_errors):.6f}")
        print(f"Median error: {np.median(prediction_errors):.6f}")
        
        # Get high error indices
        high_error_indices = model.get_high_error_indices(threshold_percentile=90)
        print(f"\nNumber of high-error states (top 10%): {len(high_error_indices)}")
        print(f"High-error state indices: {high_error_indices[:10]}...")  # Show first 10
        
        # Show highest errors
        sorted_errors = sorted(enumerate(prediction_errors), key=lambda x: x[1], reverse=True)
        print(f"\nTop 5 highest prediction errors:")
        for i, (idx, error) in enumerate(sorted_errors[:5]):
            print(f"  {i+1}. Index {idx}: Error = {error:.6f}")
    
    env.close()
    return prediction_errors


def demo_prediction_interface(model_path: str = "dqn_with_prediction"):
    """Demonstrate the prediction interface"""
    
    env = BotEvadeEnv(
        world_name="clump01_05", 
        use_lppos=False, 
        use_predator=True, 
        reward_function=custom_reward,
        max_step=300,
        time_step=0.25,
        render=False,
        real_time=False,
        action_type=BotEvadeEnv.ActionType.DISCRETE
    )
    
    # Load the trained model
    model = DQNWithPrediction.load(model_path, env=env)
    
    print("Demonstrating prediction interface...")
    
    # Reset environment
    obs, _ = env.reset()
    
    # Sample a few actions and show predictions
    for i in range(5):
        action = env.action_space.sample()
        
        # Predict next state
        predicted_next_state, confidence = model.predict_next_state(obs, action)
        
        # Actually take the action
        actual_next_obs, _, _, _, _ = env.step(action)
        
        # Calculate error
        error = model.evaluate_prediction_error(obs, action, actual_next_obs)
        
        print(f"\nStep {i+1}:")
        print(f"  Action: {action}")
        print(f"  Predicted next state: {predicted_next_state}")
        print(f"  Actual next state: {actual_next_obs}")
        print(f"  Prediction error (MSE): {error:.6f}")
        
        obs = actual_next_obs
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DQN with Prediction Training and Evaluation")
    parser.add_argument("--mode", choices=["train", "eval", "demo"], default="train",
                        help="Mode: train, eval, or demo")
    parser.add_argument("--model_path", default="dqn_with_prediction",
                        help="Path to save/load the model")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_dqn_with_prediction()
    elif args.mode == "eval":
        evaluate_prediction_errors(args.model_path)
    elif args.mode == "demo":
        demo_prediction_interface(args.model_path) 