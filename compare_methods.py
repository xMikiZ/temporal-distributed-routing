"""
Compare epsilon-greedy vs UCB action selection across distribution ratios.

For each D_r value both methods are trained from scratch.  The script saves:
  - results/avg_time_vs_dr.png   : final avg delivery time vs D_r
  - results/training_curves.png  : per-episode delivery time for a chosen D_r
  - results/summary.json         : raw numbers

Usage
-----
# Quick test (Abilene, 200 episodes):
python compare_methods.py \\
    --topo data/ABI/Topology.txt \\
    --tm_dir data/ABI/TrafficMatrix \\
    --link_weights data/ABI/link_weight.json

# Full run (more episodes, custom D_r sweep, save to custom dir):
python compare_methods.py \\
    --topo data/GEA/Topology.txt \\
    --tm_dir data/GEA/TrafficMatrix \\
    --episodes 400 \\
    --dr_values 0.0 0.2 0.4 0.6 0.8 1.0 \\
    --curve_dr 0.7 \\
    --out_dir results/geant
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

from scair.agent import IRrAgent
from scair.config import ScaIRConfig
from scair.data_loader import load_all_traffic_matrices, load_topology, normalise_tm
from scair.environment import RoutingEnvironment
from train import build_agents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_training(
    topo,
    tms,
    cfg: ScaIRConfig,
    seed: int,
) -> list:
    """Train agents with the given config and return per-episode stats."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agents = build_agents(topo, cfg)
    env = RoutingEnvironment(topo, cfg)
    history = []

    for ep in range(1, cfg.max_episodes + 1):
        tm = tms[(ep - 1) % len(tms)]
        packets = env.generate_packets(tm, cfg.packets_per_episode)
        stats = env.run_episode(packets, agents, training=True)
        history.append(stats)

        if ep == 10:
            for agent in agents:
                agent.set_learning_rate(cfg.learning_rate)
        if ep % cfg.target_update_freq == 0:
            for agent in agents:
                agent.update_target()
        if ep % cfg.sigma_decay_freq == 0:
            for agent in agents:
                agent.decay_sigma()

    return history


def last_n_mean(history: list, n: int = 20) -> float:
    """Average delivery time over the last n episodes (smoothed final perf)."""
    times = [s["avg_delivery_time"] for s in history[-n:]]
    finite = [t for t in times if t != float("inf")]
    return float(np.mean(finite)) if finite else float("inf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare epsilon-greedy vs UCB for ScaIR")
    p.add_argument("--topo", required=True, help="Topology file")
    p.add_argument("--tm_dir", required=True, help="Traffic matrix directory")
    p.add_argument("--link_weights", default=None)
    p.add_argument("--episodes", type=int, default=200,
                   help="Training episodes per (method, D_r) combination (default 200)")
    p.add_argument("--packets", type=int, default=50)
    p.add_argument("--feature_length", type=int, default=128)
    p.add_argument("--neural_units", type=int, default=64)
    p.add_argument("--ucb_c", type=float, default=2.0,
                   help="UCB exploration constant (default 2.0)")
    p.add_argument("--dr_values", type=float, nargs="+",
                   default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                   help="Distribution ratio values to sweep (default: 0 0.2 0.4 0.6 0.8 1.0)")
    p.add_argument("--curve_dr", type=float, default=None,
                   help="D_r value for which to plot full training curves "
                        "(defaults to the middle value of --dr_values)")
    p.add_argument("--out_dir", type=str, default="results",
                   help="Output directory for plots and JSON (default: results/)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- data ----
    topo = load_topology(args.topo, link_weight_file=args.link_weights)
    raw_tms = load_all_traffic_matrices(args.tm_dir, topo.num_nodes)
    tms = [normalise_tm(tm) for tm in raw_tms]

    max_deg = max(len(v) for v in topo.adjacency.values())

    def make_cfg(method: str, dr: float) -> ScaIRConfig:
        cfg = ScaIRConfig(
            feature_length=args.feature_length,
            neural_units=args.neural_units,
            max_episodes=args.episodes,
            packets_per_episode=args.packets,
            distribution_ratio=dr,
            action_method=method,
            ucb_c=args.ucb_c,
        )
        if topo.num_nodes > cfg.max_nodes:
            cfg.max_nodes = topo.num_nodes
        if max_deg > cfg.max_degree:
            cfg.max_degree = max_deg
        return cfg

    dr_values = sorted(args.dr_values)
    curve_dr = args.curve_dr if args.curve_dr is not None else dr_values[len(dr_values) // 2]

    methods = ["epsilon_greedy", "ucb"]
    labels  = {"epsilon_greedy": "ε-greedy", "ucb": f"UCB (c={args.ucb_c})"}
    colors  = {"epsilon_greedy": "tab:blue",  "ucb": "tab:orange"}

    # results[method][dr] = history list
    results: dict = {m: {} for m in methods}

    total = len(methods) * len(dr_values)
    done = 0
    for method in methods:
        for dr in dr_values:
            done += 1
            print(f"[{done}/{total}] method={method}  D_r={dr:.2f}  "
                  f"episodes={args.episodes} ...", flush=True)
            cfg = make_cfg(method, dr)
            history = run_training(topo, tms, cfg, seed=args.seed)
            results[method][dr] = history

    # ---- summary: final avg delivery time per (method, dr) ----
    summary: dict = {}
    for method in methods:
        summary[method] = {}
        for dr in dr_values:
            summary[method][str(dr)] = last_n_mean(results[method][dr])

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # ---- plot: avg delivery time vs D_r ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # --- plot 1: avg time vs D_r ---
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in methods:
            y = [last_n_mean(results[method][dr]) for dr in dr_values]
            ax.plot(dr_values, y, marker="o", label=labels[method],
                    color=colors[method], linewidth=2)
        ax.set_xlabel("Distribution ratio D_r")
        ax.set_ylabel("Avg delivery time (ms)")
        ax.set_title("ScaIR: average delivery time vs congestion level")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path1 = os.path.join(args.out_dir, "avg_time_vs_dr.png")
        fig.tight_layout()
        fig.savefig(path1, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {path1}")

        # --- plot 2: training curves for curve_dr ---
        # find closest actual dr value
        actual_dr = min(dr_values, key=lambda d: abs(d - curve_dr))
        fig, ax = plt.subplots(figsize=(10, 5))
        episodes = list(range(1, args.episodes + 1))
        for method in methods:
            times = [s["avg_delivery_time"] for s in results[method][actual_dr]]
            # replace inf with None for matplotlib gap
            times_plot = [t if t != float("inf") else None for t in times]
            ax.plot(episodes, times_plot, linewidth=0.9,
                    label=labels[method], color=colors[method])
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg delivery time (ms)")
        ax.set_title(f"Training curves at D_r = {actual_dr:.2f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path2 = os.path.join(args.out_dir, "training_curves.png")
        fig.tight_layout()
        fig.savefig(path2, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {path2}")

    except ImportError:
        print("\nmatplotlib not installed — skipping plots, see summary.json for numbers.")

    # ---- text summary ----
    print("\n" + "=" * 58)
    print(f"{'D_r':>6}  " + "  ".join(f"{labels[m]:>18}" for m in methods))
    print("-" * 58)
    for dr in dr_values:
        row = f"{dr:>6.2f}  "
        row += "  ".join(f"{summary[m][str(dr)]:>18.3f}" for m in methods)
        print(row)
    print("=" * 58)


if __name__ == "__main__":
    main()
