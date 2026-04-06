"""
PID-controlled mouse agent with a tunable trust/risk parameter.

The idea: a simple controller where we can mathematically dial
the "trust" knob and show that it directly changes trajectory shape.
This gives a known-good baseline before we test LLMs.

risk_tolerance ∈ [0, 1]:
  0 = maximally cautious (strong predator avoidance, wall-hugging)
  1 = reckless (beeline to goal, ignore predator)

partner_trust ∈ [0, 1]:
  0 = ignore partner / treat as threat
  1 = fully trust partner (willing to venture further from walls when partner nearby)

The controller blends three forces:
  1. Goal attraction (always present)
  2. Predator repulsion (scaled by 1 - risk_tolerance)
  3. Partner modulation (trust reduces effective predator fear when partner is close)
"""
import math
import numpy as np
from typing import Tuple, Optional


class PIDAgent:
    def __init__(
        self,
        risk_tolerance: float = 0.3,
        partner_trust: float = 0.0,
        goal: Tuple[float, float] = (1.0, 0.5),
        speed: float = 0.05,
    ):
        self.risk_tolerance = np.clip(risk_tolerance, 0.0, 1.0)
        self.partner_trust = np.clip(partner_trust, 0.0, 1.0)
        self.goal = np.array(goal)
        self.speed = speed

    def choose_action(
        self,
        my_pos: Tuple[float, float],
        predator_pos: Optional[Tuple[float, float]],
        partner_pos: Optional[Tuple[float, float]],
        near_wall: bool,
    ) -> Tuple[float, float, bool]:
        """Returns (target_x, target_y, wait)."""
        pos = np.array(my_pos)

        goal_vec = self.goal - pos
        goal_dist = np.linalg.norm(goal_vec)
        if goal_dist > 1e-6:
            goal_vec = goal_vec / goal_dist
        goal_force = goal_vec * 1.0

        pred_force = np.zeros(2)
        if predator_pos is not None:
            pred = np.array(predator_pos)
            diff = pos - pred
            pred_dist = np.linalg.norm(diff)
            if pred_dist > 1e-6 and pred_dist < 0.5:
                repulsion_strength = (1.0 - self.risk_tolerance) * (0.5 / max(pred_dist, 0.05)) ** 2
                if partner_pos is not None:
                    partner = np.array(partner_pos)
                    partner_dist = np.linalg.norm(pos - partner)
                    trust_reduction = self.partner_trust * max(0, 1.0 - partner_dist / 0.3)
                    repulsion_strength *= (1.0 - 0.6 * trust_reduction)
                pred_force = (diff / pred_dist) * repulsion_strength

        combined = goal_force + pred_force
        norm = np.linalg.norm(combined)
        if norm > 1e-6:
            combined = combined / norm

        target = pos + combined * self.speed
        target = np.clip(target, 0.0, 1.0)

        should_wait = False
        if predator_pos is not None:
            pred_dist = np.linalg.norm(pos - np.array(predator_pos))
            fear_threshold = 0.15 + 0.25 * self.risk_tolerance
            if partner_pos is not None:
                partner_dist = np.linalg.norm(pos - np.array(partner_pos))
                fear_threshold += 0.1 * self.partner_trust * max(0, 1 - partner_dist / 0.3)
            if pred_dist < fear_threshold and not near_wall:
                should_wait = True

        return float(target[0]), float(target[1]), should_wait

    def action_to_env(self, x, y, wait):
        return np.array([x, y, 1.0 if wait else 0.0], dtype=np.float32)
