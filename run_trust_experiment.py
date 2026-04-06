"""
Run trust experiments from YAML configs.

Usage:
    python run_trust_experiment.py --config configs/trust/sibling_vs_stranger.yaml
    python run_trust_experiment.py --config configs/trust/cross_model.yaml
    python run_trust_experiment.py --config configs/trust/trust_emergence.yaml
"""
import argparse
import json
import os
import yaml
import logging

from trust_game.dual_evade_env import DualEvadeEnv
from trust_game.state_serializer import StateSerializer
from trust_game.metrics import TrustMetrics
from trust_game.llm_agent import LLMAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_condition(cfg: dict, condition: dict, output_dir: str):
    """Run all episodes for a single experimental condition."""
    name = condition["name"]
    trust_cond = condition.get("trust_condition", "neutral")
    model1 = condition.get("model1", cfg.get("model", "google/gemini-2.5-flash"))
    model2 = condition.get("model2", model1)
    episodes = cfg.get("episodes", 5)
    max_steps = cfg.get("max_steps", 50)
    world_name = cfg.get("world_name", "21_05")
    trust_emergence = cfg.get("trust_emergence", False)

    logger.info(f"=== Condition: {name} | {model1} vs {model2} | trust={trust_cond} ===")

    env = DualEvadeEnv(
        world_name=world_name, use_predator=True,
        max_step=max_steps, time_step=0.25,
        render=False, real_time=False,
    )
    serializer = StateSerializer(trust_condition=trust_cond)
    metrics = TrustMetrics()
    last_info = None
    all_results = []

    for ep in range(episodes):
        logger.info(f"  Episode {ep+1}/{episodes}")

        a1 = LLMAgent("prey_1", model=model1, system_prompt=serializer.system_prompt("prey_1"))
        a2 = LLMAgent("prey_2", model=model2, system_prompt=serializer.system_prompt("prey_2"))
        a1.reset()
        a2.reset()

        if trust_emergence and last_info:
            a1.add_context(serializer.serialize_episode_summary(last_info, "prey_1"))
            a2.add_context(serializer.serialize_episode_summary(last_info, "prey_2"))

        obs, _ = env.reset()
        terminated = truncated = False
        ep_rewards = {"prey_1": 0.0, "prey_2": 0.0}
        step = 0

        while not terminated and not truncated:
            state = env.get_state_dict()
            x1, y1, w1, _ = a1.choose_action(serializer.serialize_state(state, "prey_1"))
            x2, y2, w2, _ = a2.choose_action(serializer.serialize_state(state, "prey_2"))
            actions = {
                "prey_1": a1.action_to_env(x1, y1, w1),
                "prey_2": a2.action_to_env(x2, y2, w2),
            }
            obs, rewards, terminated, truncated, info = env.step(actions)
            ep_rewards["prey_1"] += rewards["prey_1"]
            ep_rewards["prey_2"] += rewards["prey_2"]
            step += 1

        metrics.record_episode(info, ep_rewards)
        last_info = info
        all_results.append({
            "episode": ep + 1, "steps": step, "rewards": ep_rewards,
            "prey_1_goal": info.get("prey_1_goal_achieved", False),
            "prey_2_goal": info.get("prey_2_goal_achieved", False),
            "prey_1_puffs": info.get("prey_1_puff_count", 0),
            "prey_2_puffs": info.get("prey_2_puff_count", 0),
        })
        logger.info(f"    {step} steps | M1 goal={all_results[-1]['prey_1_goal']} | M2 goal={all_results[-1]['prey_2_goal']}")

    logger.info(f"\n{metrics.summary_table()}")

    result_path = os.path.join(output_dir, f"{name}.json")
    with open(result_path, "w") as f:
        json.dump({
            "condition": condition, "episodes": all_results,
            "metrics": {"prey_1": metrics.compute_all("prey_1"), "prey_2": metrics.compute_all("prey_2")},
        }, f, indent=2, default=str)
    logger.info(f"Saved: {result_path}")
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = os.path.join(args.output_dir, cfg.get("experiment", "unnamed"))
    os.makedirs(output_dir, exist_ok=True)

    for condition in cfg.get("conditions", []):
        run_condition(cfg, condition, output_dir)


if __name__ == "__main__":
    main()
