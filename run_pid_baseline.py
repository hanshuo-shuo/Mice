"""
PID baseline: show that trust mathematically changes trajectories.
No LLM calls. Runs instantly. Produces a 2x2 grid of trajectory plots.

Usage:
    python run_pid_baseline.py
"""
import os
import logging
import numpy as np

from trust_game.dual_evade_env import DualEvadeEnv
from trust_game.pid_agent import PIDAgent
from trust_game.metrics import TrustMetrics
from trust_game.plot import draw_arena, plot_trajectory

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

CONDITIONS = [
    {"name": "Low risk, No trust",    "risk": 0.1, "trust": 0.0},
    {"name": "Low risk, High trust",  "risk": 0.1, "trust": 1.0},
    {"name": "High risk, No trust",   "risk": 0.8, "trust": 0.0},
    {"name": "High risk, High trust", "risk": 0.8, "trust": 1.0},
]

EPISODES_PER_CONDITION = 5


def run_pid_episode(env, risk, trust):
    a1 = PIDAgent(risk_tolerance=risk, partner_trust=trust)
    a2 = PIDAgent(risk_tolerance=risk, partner_trust=trust)

    obs, _ = env.reset()
    terminated = truncated = False

    while not terminated and not truncated:
        state = env.get_state_dict()
        s1 = state["agents"]["prey_1"]
        s2 = state["agents"]["prey_2"]

        x1, y1, w1 = a1.choose_action(
            s1["position"], s1.get("predator_position"),
            s2["position"] if s1["partner_visible"] else None,
            s1["near_wall"],
        )
        x2, y2, w2 = a2.choose_action(
            s2["position"], s2.get("predator_position"),
            s1["position"] if s2["partner_visible"] else None,
            s2["near_wall"],
        )
        actions = {
            "prey_1": a1.action_to_env(x1, y1, w1),
            "prey_2": a2.action_to_env(x2, y2, w2),
        }
        obs, rewards, terminated, truncated, info = env.step(actions)

    return info


def main():
    env = DualEvadeEnv(
        world_name="21_05", use_predator=True,
        max_step=60, time_step=0.25,
        render=False, real_time=False,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, cond in enumerate(CONDITIONS):
        ax = axes[idx]
        draw_arena(ax)
        metrics = TrustMetrics()

        for ep in range(EPISODES_PER_CONDITION):
            info = run_pid_episode(env, cond["risk"], cond["trust"])
            metrics.record_episode(info)
            trajs = info.get("trajectories", {})
            alpha = 0.4 if EPISODES_PER_CONDITION > 1 else 0.9
            plot_trajectory(ax, trajs.get("prey_1", []), color="#d62728",
                           label="Mouse 1" if ep == 0 else "", alpha=alpha, linewidth=1.5)
            plot_trajectory(ax, trajs.get("prey_2", []), color="#1f77b4",
                           label="Mouse 2" if ep == 0 else "", alpha=alpha, linewidth=1.5)

        m = metrics.compute_all("prey_1")
        ax.set_title(
            f"{cond['name']}\n"
            f"wait={m['waiting_pct']:.0f}% wall={m['thigmotaxis_pct']:.0f}% "
            f"eplen={m['mean_episode_length']:.0f} solo={m['solo_venture_pct']:.0f}%",
            fontsize=10,
        )
        ax.legend(loc="lower left", fontsize=8)

    plt.suptitle("PID Baseline: Risk Tolerance × Partner Trust", fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    save_path = "results/pid_baseline_grid.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {save_path}")
    plt.show()
    env.close()


if __name__ == "__main__":
    main()
