"""
Multi-agent Gymnasium wrapper around DualEvade.
Two prey agents, one reactive predator.
Each agent gets independent observations and rewards.
"""
import trust_game._deps  # noqa: F401 — stub optional deps before cellworld imports

import sys
import os
import math
import typing
import enum
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
CELLWORLD_PATH = os.path.join(PROJECT_ROOT, "cellworld_game-main")
if CELLWORLD_PATH not in sys.path:
    sys.path.insert(0, CELLWORLD_PATH)

import cellworld_game as cwgame
from cellworld_game.tasks.dualevade import DualEvade

sys.path.insert(0, PROJECT_ROOT)
from util import find, normalize_angle, load_cell_ids_near_occlusion


STACK_FIELDS = [
    "prey_x", "prey_y", "prey_direction",
    "predator_visible", "predator_x", "predator_y", "predator_direction",
    "partner_visible", "partner_x", "partner_y",
    "near_wall", "near_occlusion",
]

ALL_FIELDS = STACK_FIELDS + [
    "puffed", "finished",
    "prey_goal_distance", "partner_distance",
    "predator_distance",
]


class DualEvadeEnv(gym.Env):
    """
    Multi-agent wrapper for the DualEvade cellworld game.
    Returns observations/rewards as dicts keyed by agent id ("prey_1", "prey_2").
    """

    class ActionType(enum.Enum):
        DISCRETE = 0
        CONTINUOUS = 1

    def __init__(
        self,
        world_name: str = "21_05",
        use_predator: bool = True,
        max_step: int = 300,
        time_step: float = 0.25,
        render: bool = False,
        real_time: bool = False,
        action_type: ActionType = ActionType.CONTINUOUS,
        frame_stack_k: int = 3,
    ):
        super().__init__()
        self.max_step = max_step
        self.time_step = time_step
        self.action_type = action_type
        self.frame_stack_k = frame_stack_k
        self.agent_ids = ["prey_1", "prey_2"]

        self.model = DualEvade(
            world_name=world_name,
            real_time=real_time,
            render=render,
            use_predator=use_predator,
            time_step=0.025,
        )
        self.loader = self.model.loader

        if world_name == "21_05":
            self.cell_ids_near_occlusion = load_cell_ids_near_occlusion(
                os.path.join(PROJECT_ROOT, "data/cell_ids_near_occlusion_21_05.npy")
            )
        else:
            self.cell_ids_near_occlusion = load_cell_ids_near_occlusion(
                os.path.join(PROJECT_ROOT, "data/cell_ids_near_occlusion.npy")
            )
        self.cell_ids_near_wall = load_cell_ids_near_occlusion(
            os.path.join(PROJECT_ROOT, "data/cell_ids_near_wall_strict.npy")
        )

        n_obs = len(STACK_FIELDS) * frame_stack_k + (len(ALL_FIELDS) - len(STACK_FIELDS))
        self.observation_space = spaces.Dict({
            aid: spaces.Box(-np.inf, np.inf, (n_obs,), dtype=np.float32)
            for aid in self.agent_ids
        })
        if action_type == self.ActionType.CONTINUOUS:
            self.action_space = spaces.Dict({
                aid: spaces.Box(0.0, 1.0, (3,), dtype=np.float32)
                for aid in self.agent_ids
            })
        else:
            n_actions = len(self.loader.full_action_list)
            self.action_space = spaces.Dict({
                aid: spaces.Discrete(n_actions) for aid in self.agent_ids
            })

        self.step_count = 0
        self.frame_stacks = {aid: deque(maxlen=frame_stack_k) for aid in self.agent_ids}
        self._trajectories = {aid: [] for aid in self.agent_ids}

    def _get_prey(self, agent_id: str):
        return self.model.prey_1 if agent_id == "prey_1" else self.model.prey_2

    def _get_prey_data(self, agent_id: str):
        return self.model.prey_data_1 if agent_id == "prey_1" else self.model.prey_data_2

    def _get_partner_id(self, agent_id: str) -> str:
        return "prey_2" if agent_id == "prey_1" else "prey_1"

    def _build_obs_vector(self, agent_id: str) -> np.ndarray:
        prey = self._get_prey(agent_id)
        prey_data = self._get_prey_data(agent_id)
        partner = self._get_prey(self._get_partner_id(agent_id))

        loc = prey.state.location
        closest_cell = find(self.loader.locations, loc[:2])
        near_wall = closest_cell in self.cell_ids_near_wall
        near_occ = closest_cell in self.cell_ids_near_occlusion

        partner_visible = self.model.mouse_visible if hasattr(self.model, "mouse_visible") else False
        partner_loc = partner.state.location if partner_visible else (0.0, 0.0)

        pred_visible = prey_data.predator_visible if self.model.use_predator else False
        pred_loc = self.model.predator.state.location if (self.model.use_predator and pred_visible) else (0.0, 0.0)
        pred_dir = normalize_angle(math.radians(self.model.predator.state.direction)) if (self.model.use_predator and pred_visible) else 0.0

        from cellworld_game.util import Point
        partner_dist = Point.distance(loc, partner.state.location)
        pred_dist = prey_data.predator_prey_distance if self.model.use_predator else 1.0

        obs = np.zeros(len(ALL_FIELDS), dtype=np.float32)
        obs[ALL_FIELDS.index("prey_x")] = loc[0]
        obs[ALL_FIELDS.index("prey_y")] = loc[1]
        obs[ALL_FIELDS.index("prey_direction")] = normalize_angle(math.radians(prey.state.direction))
        obs[ALL_FIELDS.index("predator_visible")] = float(pred_visible)
        obs[ALL_FIELDS.index("predator_x")] = pred_loc[0]
        obs[ALL_FIELDS.index("predator_y")] = pred_loc[1]
        obs[ALL_FIELDS.index("predator_direction")] = pred_dir
        obs[ALL_FIELDS.index("partner_visible")] = float(partner_visible)
        obs[ALL_FIELDS.index("partner_x")] = partner_loc[0]
        obs[ALL_FIELDS.index("partner_y")] = partner_loc[1]
        obs[ALL_FIELDS.index("near_wall")] = float(near_wall)
        obs[ALL_FIELDS.index("near_occlusion")] = float(near_occ)
        obs[ALL_FIELDS.index("puffed")] = float(prey_data.puffed)
        obs[ALL_FIELDS.index("finished")] = float(not self.model.running)
        obs[ALL_FIELDS.index("prey_goal_distance")] = prey_data.prey_goal_distance
        obs[ALL_FIELDS.index("partner_distance")] = partner_dist
        obs[ALL_FIELDS.index("predator_distance")] = pred_dist

        return obs

    def _stack_obs(self, agent_id: str, obs: np.ndarray) -> np.ndarray:
        stack_idx = [ALL_FIELDS.index(f) for f in STACK_FIELDS]
        nonstack_idx = [i for i in range(len(ALL_FIELDS)) if i not in stack_idx]

        current_stack = obs[stack_idx]
        current_nonstack = obs[nonstack_idx]

        self.frame_stacks[agent_id].append(current_stack)
        while len(self.frame_stacks[agent_id]) < self.frame_stack_k:
            self.frame_stacks[agent_id].appendleft(np.zeros_like(current_stack))

        stacked = np.concatenate(list(self.frame_stacks[agent_id]), axis=0)
        return np.concatenate([stacked, current_nonstack], axis=0)

    def _get_obs(self) -> dict:
        obs = {}
        for aid in self.agent_ids:
            raw = self._build_obs_vector(aid)
            obs[aid] = self._stack_obs(aid, raw)
        return obs

    def _get_raw_obs(self) -> dict:
        """Unstacked observations for the text serializer."""
        return {aid: self._build_obs_vector(aid) for aid in self.agent_ids}

    def _compute_rewards(self) -> dict:
        rewards = {}
        for aid in self.agent_ids:
            pd = self._get_prey_data(aid)
            r = 0.0
            if pd.puffed:
                r = -1.0
            if pd.goal_achieved:
                r = 1.0
            rewards[aid] = r
        return rewards

    def reset(self, seed=None, options=None):
        self.model.reset()
        self.step_count = 0
        for aid in self.agent_ids:
            self.frame_stacks[aid].clear()
            self._trajectories[aid] = []
        obs = self._get_obs()
        for aid in self.agent_ids:
            prey = self._get_prey(aid)
            self._trajectories[aid].append(tuple(prey.state.location[:2]))
        return obs, {}

    def step(self, actions: dict):
        for aid in self.agent_ids:
            prey = self._get_prey(aid)
            action = actions[aid]
            if self.action_type == self.ActionType.CONTINUOUS:
                if action[2] > 0.5:
                    prey.set_destination(prey.state.location[:2])
                    prey.stop_navigation()
                else:
                    prey.set_destination(tuple(action[:2]))
            else:
                prey.set_destination(self.loader.full_action_list[action])

        target_time = self.model.time + self.time_step
        while self.model.running and self.model.time < target_time:
            self.model.step()

        self.step_count += 1
        truncated = self.step_count >= self.max_step
        terminated = not self.model.running

        obs = self._get_obs()
        rewards = self._compute_rewards()

        for aid in self.agent_ids:
            prey = self._get_prey(aid)
            self._trajectories[aid].append(tuple(prey.state.location[:2]))
            pd = self._get_prey_data(aid)
            if pd.puffed:
                pd.puffed = False

        info = {
            "step": self.step_count,
            "trajectories": {aid: list(self._trajectories[aid]) for aid in self.agent_ids},
        }
        if terminated or truncated:
            for aid in self.agent_ids:
                pd = self._get_prey_data(aid)
                info[f"{aid}_puff_count"] = pd.puff_count
                info[f"{aid}_goal_achieved"] = pd.goal_achieved

        return obs, rewards, terminated, truncated, info

    def get_state_dict(self) -> dict:
        """Structured state for the text serializer (no stacking, human-readable)."""
        state = {"step": self.step_count, "agents": {}}
        for aid in self.agent_ids:
            prey = self._get_prey(aid)
            pd = self._get_prey_data(aid)
            partner = self._get_prey(self._get_partner_id(aid))
            from cellworld_game.util import Point
            state["agents"][aid] = {
                "position": (round(prey.state.location[0], 3), round(prey.state.location[1], 3)),
                "direction": round(prey.state.direction, 1),
                "goal_distance": round(pd.prey_goal_distance, 3),
                "predator_visible": pd.predator_visible if self.model.use_predator else False,
                "predator_position": (
                    round(self.model.predator.state.location[0], 3),
                    round(self.model.predator.state.location[1], 3),
                ) if (self.model.use_predator and pd.predator_visible) else None,
                "predator_distance": round(pd.predator_prey_distance, 3) if self.model.use_predator else None,
                "partner_visible": getattr(self.model, "mouse_visible", False),
                "partner_position": (
                    round(partner.state.location[0], 3),
                    round(partner.state.location[1], 3),
                ) if getattr(self.model, "mouse_visible", False) else None,
                "partner_distance": round(Point.distance(prey.state.location, partner.state.location), 3),
                "near_wall": bool(find(self.loader.locations, prey.state.location[:2]) in self.cell_ids_near_wall),
                "puffed": pd.puffed,
                "goal_achieved": pd.goal_achieved,
            }
        state["goal_location"] = (1.0, 0.5)
        return state

    def close(self):
        self.model.close()
