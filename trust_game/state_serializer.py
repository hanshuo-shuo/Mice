"""
Converts DualEvadeEnv game state into natural language for LLM agents.
"""

TRUST_PROMPTS = {
    "sibling": (
        "Mouse 2 is your sibling. You grew up together and have navigated "
        "dangerous environments side by side many times. You trust them deeply "
        "and can usually predict what they will do."
    ),
    "stranger": (
        "Mouse 2 is a stranger. You have never encountered them before. "
        "You know nothing about how they behave or whether they will help you."
    ),
    "neutral": "",
}


class StateSerializer:
    """Turns structured game state dicts into text prompts for LLM agents."""

    def __init__(self, trust_condition: str = "neutral"):
        if trust_condition not in TRUST_PROMPTS:
            raise ValueError(f"trust_condition must be one of {list(TRUST_PROMPTS.keys())}")
        self.trust_condition = trust_condition

    def system_prompt(self, agent_id: str) -> str:
        agent_num = "1" if agent_id == "prey_1" else "2"
        partner_num = "2" if agent_id == "prey_1" else "1"

        trust_text = TRUST_PROMPTS[self.trust_condition]
        if trust_text and agent_id == "prey_2":
            trust_text = trust_text.replace("Mouse 2", "Mouse 1")

        lines = [
            f"You are Mouse {agent_num} in a hexagonal arena with obstacles.",
            f"Your goal: reach the green goal point at (1.0, 0.5) without being captured.",
            f"A robotic predator hunts you. If it gets within 0.1 units, you are 'puffed' (captured).",
            f"Mouse {partner_num} is also in the arena with the same goal.",
        ]
        if trust_text:
            lines.append(trust_text)
        lines += [
            "",
            "Rules:",
            "- You can only see the predator when it is in your line of sight.",
            "- All coordinates range from 0 to 1. 1 unit = arena diameter.",
            "- Each move: provide a target (x, y) position, or choose to wait.",
            "- Your move distance per step should be small (< 0.2 units).",
            "",
            'Respond with JSON only: {"x": float, "y": float, "wait": bool, "thoughts": "..."}',
            'Set "wait": true to stay in place and observe.',
        ]
        return "\n".join(lines)

    def serialize_state(self, state_dict: dict, agent_id: str) -> str:
        agent_num = "1" if agent_id == "prey_1" else "2"
        partner_id = "prey_2" if agent_id == "prey_1" else "prey_1"
        partner_num = "2" if agent_id == "prey_1" else "1"

        me = state_dict["agents"][agent_id]
        partner = state_dict["agents"].get(partner_id, {})

        lines = [f"Step {state_dict['step']}:"]
        lines.append(f"  Your position: ({me['position'][0]}, {me['position'][1]}), facing {me['direction']}°")
        lines.append(f"  Distance to goal: {me['goal_distance']}")
        lines.append(f"  Near wall: {'yes' if me['near_wall'] else 'no'}")

        if me["predator_visible"] and me["predator_position"]:
            px, py = me["predator_position"]
            lines.append(f"  Predator: VISIBLE at ({px}, {py}), distance {me['predator_distance']}")
        else:
            lines.append("  Predator: not visible (hidden behind obstacles)")

        if me["partner_visible"] and me.get("partner_position"):
            px, py = me["partner_position"]
            lines.append(f"  Mouse {partner_num}: visible at ({px}, {py}), distance {me['partner_distance']}")
        else:
            lines.append(f"  Mouse {partner_num}: not visible, last known distance {me['partner_distance']}")

        return "\n".join(lines)

    def serialize_episode_summary(self, episode_info: dict, agent_id: str) -> str:
        """For trust-emergence experiments: summarize what the partner did last episode."""
        partner_id = "prey_2" if agent_id == "prey_1" else "prey_1"
        partner_num = "2" if agent_id == "prey_1" else "1"

        traj = episode_info.get("trajectories", {}).get(partner_id, [])
        goal = episode_info.get(f"{partner_id}_goal_achieved", False)
        puffs = episode_info.get(f"{partner_id}_puff_count", 0)

        n_steps = len(traj)
        if n_steps >= 2:
            start = traj[0]
            end = traj[-1]
            waited = sum(
                1 for i in range(1, min(7, n_steps))
                if ((traj[i][0] - traj[i-1][0])**2 + (traj[i][1] - traj[i-1][1])**2) < 0.01
            )
        else:
            start = end = (0, 0)
            waited = 0

        lines = [
            f"Previous episode summary for Mouse {partner_num}:",
            f"  Steps taken: {n_steps}",
            f"  Reached goal: {'yes' if goal else 'no'}",
            f"  Times captured: {puffs}",
            f"  Waited at start: {'yes' if waited >= 2 else 'no'} ({waited} steps with minimal movement)",
        ]
        return "\n".join(lines)
