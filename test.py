from env import BotEvadeEnv
# from env_new_rule import BotEvadeEnv
from reward import custom_reward
import cellworld_game as cwgame
import numpy as np
import matplotlib.pyplot as plt

def is_predator_visible(obs):
    """Check if predator is visible by checking if elements 45-89 are non-zero"""
    return np.any(obs[4:5] != 0)

def get_prey_position(obs):
    """Extract prey position from observation (assuming it's in the first few elements)"""
    # Assuming prey position is in first 2 elements (x, y)
    return obs[:2]

if __name__ == "__main__":
    env = BotEvadeEnv(world_name="clump01_05", 
                  use_lppos=False, 
                  use_predator=True, 
                  reward_function=custom_reward,
                  max_step=300,
                  time_step=0.25,
                  render=False,  # Turn off rendering for faster execution
                  real_time=False,
                  action_type=BotEvadeEnv.ActionType.DISCRETE,
                )
    
    # Store predator appearance positions
    predator_appearance_positions = []
    
    # Run multiple random trials
    num_trials = 3000
    
    for trial in range(num_trials):
        print(f"Running trial {trial + 1}/{num_trials}")
        
        obs, _ = env.reset()
        predator_was_visible = is_predator_visible(obs)
        
        max_steps = 20
        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            predator_is_visible = is_predator_visible(obs)
            
            # Check if predator just became visible
            if not predator_was_visible and predator_is_visible:
                prey_pos = get_prey_position(obs)
                predator_appearance_positions.append(prey_pos.copy())
                print(f"  Predator appeared at prey position: {prey_pos}")
            
            predator_was_visible = predator_is_visible
            
            if done or truncated:
                print(f"  Episode {trial + 1} finished at step {step + 1}")
                break
    
    env.close()
    
    # Convert to numpy array for easier manipulation
    predator_appearance_positions = np.array(predator_appearance_positions)
    
    print(f"\nTotal predator appearances recorded: {len(predator_appearance_positions)}")
    
    if len(predator_appearance_positions) > 0:
        # Save the positions
        np.save('predator_appearance_positions.npy', predator_appearance_positions)
        print("Positions saved to 'predator_appearance_positions.npy'")
        
        # Create density plot
        plt.figure(figsize=(10, 8))
        
        x_positions = predator_appearance_positions[:, 0]
        y_positions = predator_appearance_positions[:, 1]
        
        # Create 2D histogram (density plot)
        plt.hist2d(x_positions, y_positions, bins=20, cmap='Blues', alpha=0.01)
        plt.colorbar(label='Frequency')
        
        # Also overlay scatter plot
        plt.scatter(x_positions, y_positions, alpha=0.1, c='red', s=30, label='Predator appearances')
        
        plt.xlabel('Prey X Position')
        plt.ylabel('Prey Y Position')
        plt.title('Density Plot of Prey Positions When Predator First Appears')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig('predator_appearance_density.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        print(f"\nPosition Statistics:")
        print(f"X range: [{np.min(x_positions):.3f}, {np.max(x_positions):.3f}]")
        print(f"Y range: [{np.min(y_positions):.3f}, {np.max(y_positions):.3f}]")
        print(f"Mean position: ({np.mean(x_positions):.3f}, {np.mean(y_positions):.3f})")
        print(f"Std position: ({np.std(x_positions):.3f}, {np.std(y_positions):.3f})")
        
    else:
        print("No predator appearances detected!")