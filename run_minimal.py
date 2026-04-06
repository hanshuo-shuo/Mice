"""
Minimal trust experiment: Trust vs Enemy, one episode each, one plot.

Usage:
    python run_minimal.py
    python run_minimal.py --config configs/trust/minimal.yaml
    python run_minimal.py --model google/gemini-2.5-flash
"""
import argparse
import os
import json
import yaml
import logging
import numpy as np

from trust_game.dual_evade_env import DualEvadeEnv
from trust_game.state_serializer import StateSerializer
from trust_game.llm_agent import LLMAgent
from trust_game.metrics import TrustMetrics
from trust_game.plot import plot_dual_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/trust/minimal.yaml"


def build_system_prompt(agent_id: str, social_text: str) -> str:
    num = "1" if agent_id == "prey_1" else "2"
    partner = "2" if agent_id == "prey_1" else "1"
    return "\n".join([
        f"You are Mouse {num} in a hexagonal arena with obstacles.",
        f"Your goal: reach (1.0, 0.5) without being captured by the predator.",
        f"The predator captures you if it gets within 0.1 units.",
        f"Mouse {partner} is also in the arena.",
        "",
        social_text.strip(),
        "",
        "Rules:",
        "- You can only see the predator when it is in your line of sight.",
        "- Coordinates range from 0 to 1. Each move should be < 0.2 units.",
        "",
        'Respond ONLY with JSON: {"x": float, "y": float, "wait": bool, "thoughts": "..."}',
    ])


def run_one_episode(env, model, social_prompt, condition_name):
    """Run a single episode, return trajectories and metrics."""
    sys1 = build_system_prompt("prey_1", social_prompt)
    sys2 = build_system_prompt("prey_2", social_prompt.replace("Mouse 2", "Mouse 1"))

    agent1 = LLMAgent(agent_id="prey_1", model=model, system_prompt=sys1)
    agent2 = LLMAgent(agent_id="prey_2", model=model, system_prompt=sys2)
    serializer = StateSerializer(trust_condition="neutral")

    agent1.reset()
    agent2.reset()
    obs, _ = env.reset()

    terminated = truncated = False
    step = 0

    while not terminated and not truncated:
        state = env.get_state_dict()
        st1 = serializer.serialize_state(state, "prey_1")
        st2 = serializer.serialize_state(state, "prey_2")

        x1, y1, w1, t1 = agent1.choose_action(st1)
        x2, y2, w2, t2 = agent2.choose_action(st2)

        logger.info(
            f"  [{condition_name}] step {step}: "
            f"M1→({x1:.2f},{y1:.2f},w={w1}) M2→({x2:.2f},{y2:.2f},w={w2})"
        )

        actions = {
            "prey_1": agent1.action_to_env(x1, y1, w1),
            "prey_2": agent2.action_to_env(x2, y2, w2),
        }
        obs, rewards, terminated, truncated, info = env.step(actions)
        step += 1

    logger.info(
        f"  [{condition_name}] done: {step} steps | "
        f"M1 goal={info.get('prey_1_goal_achieved', False)} puffs={info.get('prey_1_puff_count', 0)} | "
        f"M2 goal={info.get('prey_2_goal_achieved', False)} puffs={info.get('prey_2_puff_count', 0)}"
    )
    return info.get("trajectories", {}), info, agent1.stats(), agent2.stats()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=None, help="Override model from config")
    parser.add_argument("--save", default="results/minimal_trust_vs_enemy.png")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = args.model or cfg.get("model", "google/gemini-3-flash-preview")
    max_steps = cfg.get("max_steps", 50)
    world_name = cfg.get("world_name", "21_05")
    conditions = cfg.get("conditions", [])

    if len(conditions) < 2:
        conditions = [
            {"name": "trust", "prompt_override": "Mouse 2 is your sibling. You trust them completely."},
            {"name": "enemy", "prompt_override": "Mouse 2 is dangerous. Stay away from them."},
        ]

    env = DualEvadeEnv(
        world_name=world_name,
        use_predator=True,
        max_step=max_steps,
        time_step=0.25,
        render=False,
        real_time=False,
    )

    results = {}
    for cond in conditions:
        name = cond["name"]
        prompt = cond.get("prompt_override", "")
        logger.info(f"Running condition: {name}")
        trajs, info, s1, s2 = run_one_episode(env, model, prompt, name)
        results[name] = {"trajectories": trajs, "info": info}
        logger.info(f"  API calls: M1={s1['calls']}, M2={s2['calls']}")

    env.close()

    cond_names = list(results.keys())
    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    raw_path = args.save.replace(".png", "_raw.json")
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Raw data saved to {raw_path}")

    plot_dual_comparison(
        trajs_a=results[cond_names[0]]["trajectories"],
        trajs_b=results[cond_names[1]]["trajectories"],
        title_a=cond_names[0].replace("_", " ").title(),
        title_b=cond_names[1].replace("_", " ").title(),
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
