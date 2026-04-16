"""Evaluate a random policy on the Oasis task.

Usage:
    python eval_oasis.py [--episodes 20] [--render] [--predator-ratio 0.15]
"""
import argparse
import numpy as np
from oasis_gym import OasisEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--render", default=True, action="store_true")
    p.add_argument("--predator-ratio", type=float, default=0.3,
                   help="Predator forward speed as a ratio of prey max forward speed")
    p.add_argument("--turning-ratio", type=float, default=0.175,
                   help="Predator turning speed as a ratio of prey max turning speed")
    p.add_argument("--max-step", type=int, default=600)
    p.add_argument("--no-predator", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    env = OasisEnv(
        world_name="oasis_island7_02",
        use_predator=not args.no_predator,
        predator_prey_forward_speed_ratio=args.predator_ratio,
        predator_prey_turning_speed_ratio=args.turning_ratio,
        max_step=args.max_step,
        render=args.render,
        real_time=args.render,  # only pace to real-time when rendering
    )

    results = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = truncated = False

        while not (done or truncated):
            action = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

        captures = info.get("captures", 0)
        survived = info.get("survived", 0)
        results.append({"reward": ep_reward, "captures": captures, "survived": survived})
        print(f"  ep {ep+1:3d}  reward={ep_reward:7.2f}  captures={captures}  survived={survived}")

    rewards   = [r["reward"]   for r in results]
    captures  = [r["captures"] for r in results]
    survivals = [r["survived"] for r in results]
    print("\n--- Summary ---")
    print(f"  episodes      : {args.episodes}")
    print(f"  predator ratio: {args.predator_ratio}")
    print(f"  reward   mean={np.mean(rewards):.2f}  std={np.std(rewards):.2f}")
    print(f"  captures mean={np.mean(captures):.2f}  std={np.std(captures):.2f}")
    print(f"  survival rate: {np.mean(survivals)*100:.1f}%")

    env.close()


if __name__ == "__main__":
    main()
