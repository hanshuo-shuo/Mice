from oasis_env import OasisEnv
import matplotlib.pyplot as plt
from cellworld import *
import numpy as np
import json
from PIL import Image
from cellworld_game.video import save_video_output
from openai import OpenAI
import base64
import os
import pandas as pd
import copy

# Define goal locations
goal_locations = [
    (0.75, 0.25),
    (0.75, 0.75),
    (0.25, 0.75),
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

PROMPT = """
You are controlling a prey agent (mouse) in an Oasis predator-prey environment. Your goal is to visit a sequence of goal locations in order while avoiding the predator.

Environment Details:
- You (red dot) must visit multiple goals (green dots) in sequence
- Each goal must be reached and you need to stay at the goal location briefly to complete it
- After completing a goal, a new goal will appear at the next location in the sequence
- Black areas are obstacles/walls that you cannot pass through
- The predator (blue dot) is constantly moving and trying to catch you, with a larger blue circle indicating the puffed area
- If you can't see the predator, it means it's hidden behind obstacles
- The environment has a grid to help you locate positions (x and y coordinates from 0 to 1)
- Each move must be within the valid world boundaries

Your response must be a JSON object with exactly this format:
{
  "move": [
    {"x": <float>, "y": <float>}
  ],
  "thoughts": "<single line explaining your strategy>"
}

Strategy Considerations:
1. Navigate to the current goal (green dot) efficiently
2. Avoid the predator (blue dot) when visible
3. Use obstacles strategically to hide from the predator
4. Plan your path to minimize exposure to the predator
5. Once at a goal, stay there briefly to complete it before moving to the next

Rules for moves:
1. Provide exactly 1 move (the next position to move to)
2. The move should be within valid world boundaries (approximately 0 to 1 for both x and y)
3. Move should be reachable and not through obstacles

Example response:
{
  "move": [
    {"x": 0.60, "y": 0.35}
  ],
  "thoughts": "Moving toward the current goal at (0.75, 0.25) while keeping obstacles between me and the predator."
}
"""

def get_message(image_path, PROMPT, current_goal, goals_remaining, last_action=None):
    MODEL = "gpt-4o"
    APIKEY = ### YOUR OPENAI API KEY HERE ###
    client = OpenAI(api_key=APIKEY)

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    base64_image = encode_image(image_path)

    messages = [
        {"role": "system", "content": PROMPT}
    ]

    # Add current goal and goals remaining information
    goal_info = f"Current goal location: ({current_goal[0]:.2f}, {current_goal[1]:.2f}). Goals remaining: {goals_remaining}"
    messages.append({
        "role": "system",
        "content": goal_info
    })

    if last_action:
        messages.append({
            "role": "system",
            "content": f"Your last action was: {last_action}"
        })

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "low"
                }
            }
        ]
    })

    completion = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=messages
    )
    return completion.choices[0].message.content

def draw_oasis_frame(obs, all_goal_locations):
    """
    Draw the Oasis environment frame with prey, predator, current goal, and all goal locations.

    Args:
        obs: OasisObservation object with fields like prey_x, prey_y, predator_x, predator_y, goal_x, goal_y, etc.
        all_goal_locations: List of all goal locations in the sequence
    """
    world = World.get_from_parameters_names("hexagonal", "canonical", "21_05")
    Display(world, show_axes=True, animated=True)

    # Plot the prey (you)
    plt.plot(obs[0], obs[1], 'o', markersize=15, color='r')
    plt.text(obs[0], obs[1], 'you', fontsize=12, verticalalignment='bottom', color='r')

    # Plot current goal
    plt.plot(obs[6], obs[7], 'o', markersize=15, color='g')
    plt.gca().add_patch(plt.Circle((obs[6], obs[7]), 0.05, color='g', fill=True, alpha=0.5))
    plt.text(obs[6], obs[7], f'goal', fontsize=12, verticalalignment='bottom', color='g')

    # Plot all goal locations as small markers
    for i, goal_loc in enumerate(all_goal_locations):
        plt.plot(goal_loc[0], goal_loc[1], 's', markersize=8, color='lightgreen', alpha=0.5)

    # Check if predator is visible and plot if so
    if obs[3] != 0 or obs[4] != 0:  # predator_x, predator_y
        predator_exists = True
        plt.plot(obs[3], obs[4], 'o', markersize=15, color='b')
        plt.text(obs[3], obs[4], 'predator', fontsize=12, verticalalignment='bottom', color='b')
        plt.gca().add_patch(plt.Circle((obs[3], obs[4]), 0.1, color='b', fill=True, alpha=0.3))
    else:
        predator_exists = False

    # Add title with goals remaining
    goals_remaining = int(obs[13])  # goals_remaining field
    title_text = f"Oasis Environment - Goals Remaining: {goals_remaining}"
    if not predator_exists:
        title_text += " (Predator hidden)"
    plt.text(0, 1, title_text, fontsize=12, verticalalignment='bottom', color='black')

    # Grid and axes
    plt.xticks(np.arange(0, 1.05, 0.05))
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.grid(linestyle='--', linewidth=0.5)
    fig = plt.gcf()
    return fig

def eval_llm():
    # Create directories for storing outputs
    os.makedirs("oasis_frames", exist_ok=True)
    os.makedirs("oasis_god_view", exist_ok=True)
    os.makedirs("oasis_video", exist_ok=True)

    # Create environment
    env = OasisEnv(
        world_name="21_05",
        goal_locations=goal_locations,
        use_lppos=False,
        use_predator=True,
        max_step=500,
        time_step=0.25,
        reward_function=my_reward_fn,
        render=True,
        real_time=False,
        action_type=OasisEnv.ActionType.CONTINUOUS
    )

    num_episodes = 1
    for episode in range(1, num_episodes + 1):
        obs_list = []
        action_list = []
        reward_list = []
        done_list = []
        next_obs_list = []
        thoughts_list = []

        obs, _ = env.reset()

        # Draw initial frame
        fig = draw_oasis_frame(obs, goal_locations)
        fig.savefig(f"oasis_frames/episode_{episode}_frame_0.png", dpi=64)
        plt.close(fig)

        # Save god view
        temp_frame = env.model.view.get_screen()
        temp_frame = Image.fromarray(temp_frame)
        temp_frame.save(f"oasis_god_view/episode_{episode}_frame_0.png")
        save_video_output(env.model, f"oasis_video/episode_{episode}")

        step = 1
        last_action = None
        done = False
        truncated = False

        # Create episode-specific thought file
        with open(f"oasis_thought_episode_{episode}.txt", "w") as f:
            f.write("")

        print(f"\n=== Episode {episode} Started ===")
        print(f"Goal sequence: {goal_locations}")

        while not (done or truncated):
            fig = draw_oasis_frame(obs, goal_locations)
            temp_frame_path = f"oasis_frames/episode_{episode}_frame_{step}.png"
            fig.savefig(temp_frame_path, dpi=64)
            plt.close(fig)

            # Get current goal and goals remaining
            current_goal = (obs[6], obs[7])  # goal_x, goal_y
            goals_remaining = int(obs[13])  # goals_remaining

            try:
                response = json.loads(get_message(temp_frame_path, PROMPT, current_goal, goals_remaining, last_action))
            except Exception as e:
                print(f"Error occurred in episode {episode}, step {step}: {e}")
                break

            print(f"Step {step}: {response}")
            thoughts = response["thoughts"]
            action = response["move"][0]

            # Store current state
            obs_list.append(copy.deepcopy(obs))
            action_list.append([action["x"], action["y"]])
            thoughts_list.append(thoughts)

            # Take action
            next_obs, reward, done, truncated, info = env.step(action=np.array([action["x"], action["y"]]))

            # Store transition
            reward_list.append(reward)
            done_list.append(done or truncated)
            next_obs_list.append(copy.deepcopy(next_obs))

            # Print progress
            if obs[10] > 0.5:  # goal_just_completed
                print(f"  --> Goal completed! Goals remaining: {goals_remaining - 1}")
            if obs[11] > 0.5:  # puffed
                print(f"  --> Puffed by predator!")

            obs = next_obs

            # Draw updated frame
            fig = draw_oasis_frame(obs, goal_locations)
            fig.savefig(f"oasis_frames/episode_{episode}_frame_{step}.png", dpi=64)
            plt.close(fig)

            # Save god view
            temp_frame = env.model.view.get_screen()
            temp_frame = Image.fromarray(temp_frame)
            temp_frame.save(f"oasis_god_view/episode_{episode}_frame_{step}.png")

            last_action = json.dumps(response["move"])

            with open(f"oasis_thought_episode_{episode}.txt", "a") as f:
                f.write(f"Step {step}: {thoughts}\n")

            step += 1

        # Episode summary
        total_reward = sum(reward_list)
        puff_count = info.get("puff_count", 0)
        goals_completed = len(goal_locations) - len(env.model.goal_sequence)
        is_success = info.get("is_success", 0)

        print(f"\n=== Episode {episode} Completed ===")
        print(f"Total steps: {step - 1}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Goals completed: {goals_completed}/{len(goal_locations)}")
        print(f"Puff count: {puff_count}")
        print(f"Success: {is_success}")

        # Save episode data
        data = {
            "obs": obs_list,
            "action": action_list,
            "reward": reward_list,
            "done": done_list,
            "next_obs": next_obs_list,
            "thoughts": thoughts_list
        }
        df = pd.DataFrame(data)
        df.to_csv(f"oasis_trajectory_data_episode_{episode}.csv", index=False)

    env.close()

if __name__ == "__main__":
    eval_llm()