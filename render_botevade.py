"""
render_botevade.py
------------------
Interactive render demo for BotEvadeEnv with random policy.

Usage:
    python render_botevade.py
    python render_botevade.py --no-predator
    python render_botevade.py --continuous
    python render_botevade.py --episodes 3
"""

import argparse
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CELLWORLD_PATH = os.path.join(BASE_DIR, "cellworld_game-main")
if CELLWORLD_PATH not in sys.path:
    sys.path.insert(0, CELLWORLD_PATH)

from botevade_gym import BotEvadeEnv


def run(args):
    action_type = (
        BotEvadeEnv.ActionType.CONTINUOUS if args.continuous else BotEvadeEnv.ActionType.DISCRETE
    )

    env = BotEvadeEnv(
        world_name="21_05",
        use_predator=not args.no_predator,
        use_lppos=False,
        render=True,
        real_time=True,
        action_type=action_type,
        time_step=0.25,
        max_step=2000,
    )

    print(f"BotEvadeEnv created")
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

        print(f"[Episode {episode + 1}] Starting")

        while not (done or truncated):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

        status = "DONE" if done else "TRUNCATED"
        puffs = info.get("captures", 0)
        print(f"[Episode {episode + 1}] {status} | "
              f"steps={step} | puffs={puffs} | reward={total_reward:.2f}")
        print()

    env.close()
    print("Render session finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render BotEvadeEnv with random policy")
    parser.add_argument("--episodes", type=int, default=2,
                        help="Number of episodes to run (default: 2)")
    parser.add_argument("--no-predator", action="store_true",
                        help="Disable the predator robot")
    parser.add_argument("--continuous", action="store_true",
                        help="Use continuous action space instead of discrete")
    args = parser.parse_args()
    run(args)
