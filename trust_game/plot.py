"""
Plot trajectories on the cellworld arena.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from typing import List, Tuple, Optional, Dict


ARENA_CENTER = (0.5, 0.5)
ARENA_RADIUS = 0.5
GOAL = (1.0, 0.5)


def _hexagon_vertices(cx, cy, r, angle_deg=0):
    verts = []
    for i in range(6):
        a = math.radians(60 * i + angle_deg)
        verts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return verts


def draw_arena(ax, occlusion_centers=None, occlusion_radius=0.04):
    hex_verts = _hexagon_vertices(0.5, 0.5, 0.5, angle_deg=0)
    arena = plt.Polygon(hex_verts, fill=True, facecolor="#e8e8e8",
                        edgecolor="black", linewidth=1.5, zorder=0)
    ax.add_patch(arena)

    if occlusion_centers:
        for cx, cy in occlusion_centers:
            occ = plt.Polygon(
                _hexagon_vertices(cx, cy, occlusion_radius, angle_deg=0),
                fill=True, facecolor="#3a3a3a", edgecolor="#2a2a2a",
                linewidth=0.5, zorder=5
            )
            ax.add_patch(occ)

    ax.plot(*GOAL, marker="*", color="green", markersize=14, zorder=10)
    ax.plot(0.05, 0.5, marker="o", color="gray", markersize=8, zorder=10)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_trajectory(ax, traj: List[Tuple[float, float]], color="red",
                    label="", alpha=0.8, linewidth=2):
    if not traj:
        return
    xs, ys = zip(*traj)
    ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth,
            label=label, zorder=20)
    ax.plot(xs[0], ys[0], "o", color=color, markersize=6, zorder=25)
    ax.plot(xs[-1], ys[-1], "s", color=color, markersize=6, zorder=25)


def plot_dual_comparison(
    trajs_a: Dict[str, List[Tuple[float, float]]],
    trajs_b: Dict[str, List[Tuple[float, float]]],
    title_a: str = "Trust",
    title_b: str = "Enemy",
    save_path: Optional[str] = None,
    occlusion_centers=None,
):
    """Side-by-side plot of two conditions, each with two mice trajectories."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    draw_arena(ax1, occlusion_centers)
    draw_arena(ax2, occlusion_centers)

    plot_trajectory(ax1, trajs_a.get("prey_1", []), color="#d62728", label="Mouse 1")
    plot_trajectory(ax1, trajs_a.get("prey_2", []), color="#1f77b4", label="Mouse 2")
    ax1.set_title(title_a, fontsize=14, fontweight="bold")
    ax1.legend(loc="lower left", fontsize=9)

    plot_trajectory(ax2, trajs_b.get("prey_1", []), color="#d62728", label="Mouse 1")
    plot_trajectory(ax2, trajs_b.get("prey_2", []), color="#1f77b4", label="Mouse 2")
    ax2.set_title(title_b, fontsize=14, fontweight="bold")
    ax2.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()
    return fig
