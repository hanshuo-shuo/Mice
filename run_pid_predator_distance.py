"""
PID baseline: average distance to predator under low vs high trust.

Hypothesis: high-trust mice should sit *closer* to the predator on average
(riskier behavior) than low-trust mice. We use the same PID controller as
run_pid_baseline.py and only vary the partner_trust knob.

No LLM calls. Runs instantly. Produces a 2x2 grid:
  (0,0) Mouse 1 distance to predator over time, low vs high trust
  (0,1) Mouse 2 distance to predator over time, low vs high trust
  (1,0) Combined per-step mean distance (with std band)
  (1,1) Bar chart of overall mean distance per condition

Usage:
    python run_pid_predator_distance.py
"""
import os
import math
import logging
import numpy as np

from trust_game.dual_evade_env import DualEvadeEnv
from trust_game.pid_agent import PIDAgent

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


CONDITIONS = [
    {"name": "Low trust",  "trust": 0.0, "color": "#d62728"},
    {"name": "High trust", "trust": 1.0, "color": "#1f77b4"},
]
RISK_TOLERANCE = 0.5
EPISODES_PER_CONDITION = 20


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def run_pid_episode(env, risk, trust):
    """Run one PID episode and return per-step distances from each prey to the predator."""
    a1 = PIDAgent(risk_tolerance=risk, partner_trust=trust)
    a2 = PIDAgent(risk_tolerance=risk, partner_trust=trust)

    env.reset()
    terminated = truncated = False

    d1, d2 = [], []
    # Initial distances at reset
    pred_loc = tuple(env.model.predator.state.location[:2])
    p1 = tuple(env.model.prey_1.state.location[:2])
    p2 = tuple(env.model.prey_2.state.location[:2])
    d1.append(_dist(p1, pred_loc))
    d2.append(_dist(p2, pred_loc))

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
        _, _, terminated, truncated, _ = env.step(actions)

        pred_loc = tuple(env.model.predator.state.location[:2])
        p1 = tuple(env.model.prey_1.state.location[:2])
        p2 = tuple(env.model.prey_2.state.location[:2])
        d1.append(_dist(p1, pred_loc))
        d2.append(_dist(p2, pred_loc))

    return np.array(d1), np.array(d2)


def _pad_to(arr_list, length, fill=np.nan):
    """Pad each 1D array to the same length with NaN so we can take a nan-mean."""
    out = np.full((len(arr_list), length), fill, dtype=float)
    for i, a in enumerate(arr_list):
        n = min(len(a), length)
        out[i, :n] = a[:n]
    return out


def main():
    env = DualEvadeEnv(
        world_name="21_05", use_predator=True,
        max_step=60, time_step=0.25,
        render=False, real_time=False,
    )

    results = {}  # cond name -> dict with mouse1, mouse2 lists of arrays
    for cond in CONDITIONS:
        logger.info(f"Running condition: {cond['name']} (trust={cond['trust']})")
        m1_runs, m2_runs = [], []
        for ep in range(EPISODES_PER_CONDITION):
            d1, d2 = run_pid_episode(env, RISK_TOLERANCE, cond["trust"])
            m1_runs.append(d1)
            m2_runs.append(d2)
            logger.info(f"  ep {ep+1}/{EPISODES_PER_CONDITION}: "
                        f"mean_d1={d1.mean():.3f} mean_d2={d2.mean():.3f}")
        results[cond["name"]] = {"m1": m1_runs, "m2": m2_runs, "color": cond["color"]}

    env.close()

    # Align all runs to the longest length seen
    max_len = max(
        max(len(a) for a in r["m1"] + r["m2"])
        for r in results.values()
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- (0,0) Mouse 1: distance over time ----
    ax = axes[0, 0]
    for name, r in results.items():
        padded = _pad_to(r["m1"], max_len)
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        t = np.arange(max_len)
        ax.plot(t, mean, color=r["color"], label=name, linewidth=2)
        ax.fill_between(t, mean - std, mean + std, color=r["color"], alpha=0.15)
    ax.set_title("Mouse 1: distance to predator over time", fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to predator")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- (0,1) Mouse 2: distance over time ----
    ax = axes[0, 1]
    for name, r in results.items():
        padded = _pad_to(r["m2"], max_len)
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        t = np.arange(max_len)
        ax.plot(t, mean, color=r["color"], label=name, linewidth=2)
        ax.fill_between(t, mean - std, mean + std, color=r["color"], alpha=0.15)
    ax.set_title("Mouse 2: distance to predator over time", fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to predator")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- (1,0) Combined (both mice) mean distance over time ----
    ax = axes[1, 0]
    for name, r in results.items():
        padded = _pad_to(r["m1"] + r["m2"], max_len)
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        t = np.arange(max_len)
        ax.plot(t, mean, color=r["color"], label=name, linewidth=2)
        ax.fill_between(t, mean - std, mean + std, color=r["color"], alpha=0.15)
    ax.set_title("Both mice combined: mean distance to predator over time", fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to predator")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- (1,1) Bar chart: overall mean per condition ----
    ax = axes[1, 1]
    labels, means, errs, colors = [], [], [], []
    for name, r in results.items():
        per_episode = []
        for d1, d2 in zip(r["m1"], r["m2"]):
            per_episode.append(np.concatenate([d1, d2]).mean())
        labels.append(name)
        means.append(np.mean(per_episode))
        errs.append(np.std(per_episode) / math.sqrt(len(per_episode)))
        colors.append(r["color"])
    xs = np.arange(len(labels))
    ax.bar(xs, means, yerr=errs, color=colors, alpha=0.8, capsize=8,
           edgecolor="black")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean distance to predator (whole trial)")
    ax.set_title(f"Overall mean across {EPISODES_PER_CONDITION} episodes  (±SEM)",
                 fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    for x, m, e in zip(xs, means, errs):
        ax.text(x, m + e + 0.005, f"{m:.3f}", ha="center", fontsize=10)

    plt.suptitle(
        f"PID Baseline — Distance to Predator (risk={RISK_TOLERANCE}, "
        f"{EPISODES_PER_CONDITION} episodes/condition)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    save_path = "results/pid_predator_distance.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {save_path}")

    # Print summary
    print("\n=== Summary ===")
    for name, r in results.items():
        all_d = np.concatenate([np.concatenate([d1, d2]) for d1, d2 in zip(r["m1"], r["m2"])])
        print(f"{name:12s}  mean={all_d.mean():.3f}  median={np.median(all_d):.3f}  "
              f"min={all_d.min():.3f}")


if __name__ == "__main__":
    main()
