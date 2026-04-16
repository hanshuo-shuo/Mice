"""Destination-tracking PID controller for a point-mass Mouse.

Takes a 2D waypoint and produces `(ax, ay) ∈ [-1, 1]²` to drive the prey's
`PointDynamics` toward it. This replaces the old in-`NavigationAgent` PID
that spoke unicycle (forward_speed + turn_speed) and used A* path planning.

Usage:

    ctrl = MousePIDController(prey)
    ctrl.set_destination((x, y))
    while env.running:
        ctrl.step()      # reads prey.state, calls prey.set_action(ax, ay)
        env.step()

Or one-shot:

    ax, ay = MousePIDController.track(prey, destination)
    prey.set_action(ax, ay)
"""
from typing import Optional, Tuple

import numpy as np

from .mouse import Mouse


class MousePIDController:
    def __init__(self,
                 mouse: Mouse,
                 approach_speed: Optional[float] = None,
                 kp_v: float = 5.0,
                 arrival_radius: float = 0.02):
        """
        Parameters
        ----------
        mouse : Mouse
            The prey whose `set_action` we drive.
        approach_speed : float, optional
            Desired cruise speed. Defaults to `mouse.max_forward_speed`.
        kp_v : float
            Proportional gain on velocity error. Higher = snappier response
            to direction changes but more overshoot.
        arrival_radius : float
            Within this distance of the destination we command zero velocity
            (let damping stop the prey).
        """
        self.mouse = mouse
        self.approach_speed = (approach_speed if approach_speed is not None
                               else mouse.max_forward_speed)
        self.kp_v = kp_v
        self.arrival_radius = arrival_radius
        self.destination: Optional[Tuple[float, float]] = None

    def set_destination(self, destination: Tuple[float, float]) -> None:
        self.destination = (float(destination[0]), float(destination[1]))

    def clear(self) -> None:
        self.destination = None
        self.mouse.set_action(0.0, 0.0)

    def step(self) -> Tuple[float, float]:
        """Compute `(ax, ay)` for the current state and apply it to the
        prey. Returns the applied action so callers can log it."""
        if self.destination is None:
            self.mouse.set_action(0.0, 0.0)
            return 0.0, 0.0
        ax, ay = self._compute(self.mouse, self.destination,
                               self.approach_speed, self.kp_v,
                               self.arrival_radius)
        self.mouse.set_action(ax, ay)
        return ax, ay

    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute(mouse: Mouse,
                 destination: Tuple[float, float],
                 approach_speed: float,
                 kp_v: float,
                 arrival_radius: float) -> Tuple[float, float]:
        px, py = mouse.state.location
        vx, vy = mouse.state.velocity
        dx = destination[0] - px
        dy = destination[1] - py
        dist = float(np.hypot(dx, dy))

        if dist < arrival_radius:
            # command zero velocity; let damping decelerate us
            desired_vx = 0.0
            desired_vy = 0.0
        else:
            # ease down the commanded speed as we approach the target so we
            # don't blow through it (simple linear ramp over 2x arrival)
            ramp = min(1.0, dist / (2.0 * arrival_radius + 1e-6))
            speed_cmd = approach_speed * ramp
            desired_vx = (dx / dist) * speed_cmd
            desired_vy = (dy / dist) * speed_cmd

        ax = kp_v * (desired_vx - vx)
        ay = kp_v * (desired_vy - vy)
        # clip to action bounds; the dynamics will scale by accel_scale
        ax = float(np.clip(ax, -1.0, 1.0))
        ay = float(np.clip(ay, -1.0, 1.0))
        return ax, ay

    @classmethod
    def track(cls,
              mouse: Mouse,
              destination: Tuple[float, float],
              approach_speed: Optional[float] = None,
              kp_v: float = 5.0,
              arrival_radius: float = 0.02) -> Tuple[float, float]:
        """Stateless helper: compute `(ax, ay)` without keeping a controller
        object around."""
        speed = (approach_speed if approach_speed is not None
                 else mouse.max_forward_speed)
        return cls._compute(mouse, destination, speed, kp_v, arrival_radius)
