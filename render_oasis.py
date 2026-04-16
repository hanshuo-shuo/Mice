"""
render_oasis.py
---------------
Interactive render demo for OasisEnv.

The prey uses a simple heuristic: navigate toward the current active goal
by picking the action whose destination is closest to the goal location.
This lets you visually verify that the environment runs, renders, and
transitions goals correctly.

Usage:
    python render_oasis.py
    python render_oasis.py --no-predator
    python render_oasis.py --continuous
    python render_oasis.py --episodes 3
"""

import argparse
import sys
import os
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CELLWORLD_PATH = os.path.join(BASE_DIR, "cellworld_game-main")
if CELLWORLD_PATH not in sys.path:
    sys.path.insert(0, CELLWORLD_PATH)

import numpy as np
from oasis_gym import OasisEnv


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def greedy_discrete_action(env: OasisEnv) -> int:
    """Pick the action (destination) closest to the current goal."""
    goal = env.model.goal_location
    if goal is None:
        return 0
    best_idx = 0
    best_d = float("inf")
    for i, loc in enumerate(env.action_list):
        d = dist(loc, goal)
        if d < best_d:
            best_d = d
            best_idx = i
    return best_idx


def greedy_continuous_action(env: OasisEnv) -> np.ndarray:
    """Return the goal location directly as the continuous action."""
    goal = env.model.goal_location
    if goal is None:
        return np.array([0.5, 0.5], dtype=np.float32)
    return np.array(goal, dtype=np.float32)


def run(args):
    action_type = (
        OasisEnv.ActionType.CONTINUOUS if args.continuous else OasisEnv.ActionType.DISCRETE
    )

    env = OasisEnv(
        world_name="oasis_island7_02",
        use_predator=not args.no_predator,
        render=True,
        real_time=True,
        action_type=action_type,
        time_step=0.25,        
        max_step=2000,
        puff_cool_down_time=0.5,
        puff_threshold=0.1,
        goal_threshold=0.025,
        goal_time=1.0,
    )

    print(f"OasisEnv created")
    print(f"  Observation space : {env.observation_space}")
    print(f"  Action space      : {env.action_space}")
    print(f"  Action type       : {action_type.name}")
    print(f"  Use predator      : {not args.no_predator}")
    print()

    for episode in range(args.episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        step = 0
        goals_hit = 0
        prev_goal = env.model.goal_location

        print(f"[Episode {episode + 1}] Starting — goal sequence length: "
              f"{len(env.model.goal_sequence) + 1}")

        while not (done or truncated):
            action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            # Detect goal change
            current_goal = env.model.goal_location
            if current_goal != prev_goal:
                goals_hit += 1
                print(f"  Step {step:4d}: goal reached! "
                      f"Goals remaining: {int(obs[-1])}")
                prev_goal = current_goal

        status = "DONE" if done else "TRUNCATED"
        print(f"[Episode {episode + 1}] {status} | "
              f"steps={step} | goals_hit={goals_hit} | "
              f"puffs={env.model.puff_count} | reward={total_reward:.2f}")
        print()

    env.close()
    print("Render session finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render OasisEnv with a greedy agent")
    parser.add_argument("--episodes", type=int, default=2,
                        help="Number of episodes to run (default: 2)")
    parser.add_argument("--no-predator", action="store_true",
                        help="Disable the predator robot")
    parser.add_argument("--continuous", action="store_true",
                        help="Use continuous action space instead of discrete")
    args = parser.parse_args()
    run(args)
