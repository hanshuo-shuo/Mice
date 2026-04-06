"""
Behavioral metrics for comparing agent trajectories.
Mirrors the metrics from the ICML paper + multi-agent extensions.
"""
import math
import numpy as np
from typing import List, Tuple, Dict, Optional


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def _dist_to_wall(point: Tuple[float, float], arena_center=(0.5, 0.5), arena_radius=0.5) -> float:
    return arena_radius - _dist(point, arena_center)


class TrustMetrics:
    """Computes behavioral metrics over completed episodes."""

    WALL_THRESHOLD = 0.1
    WAITING_MOVEMENT_THRESHOLD = 0.1
    WAITING_STEPS = 6

    def __init__(self):
        self.episodes: List[dict] = []

    def record_episode(self, info: dict, rewards: Optional[dict] = None):
        self.episodes.append({
            "info": info,
            "rewards": rewards or {},
        })

    def compute_all(self, agent_id: str = "prey_1") -> dict:
        if not self.episodes:
            return {}
        return {
            "waiting_pct": self.waiting_percentage(agent_id),
            "thigmotaxis_pct": self.thigmotaxis_percentage(agent_id),
            "mean_episode_length": self.mean_episode_length(agent_id),
            "mean_predator_proximity": self.mean_predator_proximity(agent_id),
            "survival_rate": self.survival_rate(agent_id),
            "goal_rate": self.goal_rate(agent_id),
            "mean_partner_distance": self.mean_partner_distance(agent_id),
            "solo_venture_pct": self.solo_venture_percentage(agent_id),
        }

    def _get_trajectories(self, agent_id: str) -> List[List[Tuple[float, float]]]:
        trajs = []
        for ep in self.episodes:
            t = ep["info"].get("trajectories", {}).get(agent_id, [])
            if t:
                trajs.append(t)
        return trajs

    def waiting_percentage(self, agent_id: str) -> float:
        """% of episodes where agent barely moves in first N steps."""
        trajs = self._get_trajectories(agent_id)
        if not trajs:
            return 0.0
        waiting = 0
        for traj in trajs:
            n = min(self.WAITING_STEPS + 1, len(traj))
            total_movement = sum(
                _dist(traj[i], traj[i-1]) for i in range(1, n)
            )
            if total_movement < self.WAITING_MOVEMENT_THRESHOLD:
                waiting += 1
        return waiting / len(trajs) * 100

    def thigmotaxis_percentage(self, agent_id: str) -> float:
        """Average % of trajectory points within WALL_THRESHOLD of the arena wall."""
        trajs = self._get_trajectories(agent_id)
        if not trajs:
            return 0.0
        pcts = []
        for traj in trajs:
            near = sum(1 for p in traj if _dist_to_wall(p) < self.WALL_THRESHOLD)
            pcts.append(near / len(traj) * 100)
        return float(np.mean(pcts))

    def mean_episode_length(self, agent_id: str) -> float:
        trajs = self._get_trajectories(agent_id)
        if not trajs:
            return 0.0
        return float(np.mean([len(t) for t in trajs]))

    def mean_predator_proximity(self, agent_id: str) -> float:
        """Placeholder — requires predator positions stored per-step.
        For now, reports NaN. To be extended with per-step predator logging."""
        return float("nan")

    def survival_rate(self, agent_id: str) -> float:
        """% of episodes where agent was not puffed."""
        if not self.episodes:
            return 0.0
        survived = sum(
            1 for ep in self.episodes
            if ep["info"].get(f"{agent_id}_puff_count", 0) == 0
        )
        return survived / len(self.episodes) * 100

    def goal_rate(self, agent_id: str) -> float:
        if not self.episodes:
            return 0.0
        reached = sum(
            1 for ep in self.episodes
            if ep["info"].get(f"{agent_id}_goal_achieved", False)
        )
        return reached / len(self.episodes) * 100

    def mean_partner_distance(self, agent_id: str) -> float:
        """Average distance between the two agents across all trajectory steps."""
        partner_id = "prey_2" if agent_id == "prey_1" else "prey_1"
        dists = []
        for ep in self.episodes:
            t1 = ep["info"].get("trajectories", {}).get(agent_id, [])
            t2 = ep["info"].get("trajectories", {}).get(partner_id, [])
            n = min(len(t1), len(t2))
            for i in range(n):
                dists.append(_dist(t1[i], t2[i]))
        return float(np.mean(dists)) if dists else 0.0

    def solo_venture_percentage(self, agent_id: str) -> float:
        """% of trajectory steps where the agent is >0.3 units from partner."""
        partner_id = "prey_2" if agent_id == "prey_1" else "prey_1"
        solo = 0
        total = 0
        for ep in self.episodes:
            t1 = ep["info"].get("trajectories", {}).get(agent_id, [])
            t2 = ep["info"].get("trajectories", {}).get(partner_id, [])
            n = min(len(t1), len(t2))
            for i in range(n):
                total += 1
                if _dist(t1[i], t2[i]) > 0.3:
                    solo += 1
        return (solo / total * 100) if total > 0 else 0.0

    def summary_table(self) -> str:
        lines = ["Agent        | Wait% | Wall% | EpLen | Survive% | Goal% | PartnerDist | Solo%"]
        lines.append("-" * 80)
        for aid in ["prey_1", "prey_2"]:
            m = self.compute_all(aid)
            if not m:
                continue
            lines.append(
                f"{aid:12s} | {m['waiting_pct']:5.1f} | {m['thigmotaxis_pct']:5.1f} | "
                f"{m['mean_episode_length']:5.1f} | {m['survival_rate']:8.1f} | {m['goal_rate']:5.1f} | "
                f"{m['mean_partner_distance']:11.3f} | {m['solo_venture_pct']:5.1f}"
            )
        return "\n".join(lines)
