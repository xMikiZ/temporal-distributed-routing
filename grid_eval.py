#!/usr/bin/env python3
"""
Grid evaluation: ScaIR vs OSPF across D_r x n_packets combinations.

Evaluates a trained checkpoint under varying traffic conditions to identify
where ScaIR's adaptive routing outperforms the static OSPF baseline.

Produces two plots:
  1. Line plots: delivery time vs D_r, one subplot per packet count.
  2. Heatmap: ScaIR - OSPF difference across the full grid.
"""

import argparse
import random
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scair_delay.config import ScaIRConfig
from scair_delay.data_loader import load_topology, load_all_traffic_matrices, normalise_tm
from scair_delay.environment import RoutingEnvironment
from train_delay import build_agents, load_checkpoint
from evaluate_delay import run_ospf


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_scair(topo, cfg, tms, agents, episodes, n_packets):
    env = RoutingEnvironment(topo, cfg)
    times = []
    for ep in range(episodes):
        tm = tms[ep % len(tms)]
        pkts = env.generate_packets(tm, n_packets)
        stats = env.run_episode(pkts, agents, training=False)
        times.append(stats["avg_delivery_time"])
    return float(np.mean(times))


def eval_ospf(topo, cfg, tms, episodes, n_packets):
    times = []
    for ep in range(episodes):
        tm = tms[ep % len(tms)]
        stats = run_ospf(topo, cfg, tm, n_packets)
        times.append(stats["avg_delivery_time"])
    return float(np.mean(times))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Grid evaluation: ScaIR vs OSPF")
    p.add_argument("--topo", required=True)
    p.add_argument("--tm_dir", required=True)
    p.add_argument("--link_weights", default=None)
    p.add_argument("--checkpoint", default=None,
                   help="Trained checkpoint to evaluate (optional; random init if omitted)")
    p.add_argument("--delay_input", action="store_true")
    p.add_argument("--delay_init", action="store_true")
    p.add_argument("--dr_values", nargs="+", type=float,
                   default=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                   help="D_r values to sweep")
    p.add_argument("--packet_counts", nargs="+", type=int,
                   default=[50, 100, 200],
                   help="Packet counts per episode to sweep")
    p.add_argument("--episodes", type=int, default=50,
                   help="Evaluation episodes per grid cell (default 50)")
    p.add_argument("--feature_length", type=int, default=128)
    p.add_argument("--neural_units", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_plot", default="results/grid_eval.png",
                   help="Output path for the line-plot figure")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading topology: {args.topo}")
    topo = load_topology(args.topo, link_weight_file=args.link_weights)
    raw_tms = load_all_traffic_matrices(args.tm_dir, topo.num_nodes)
    tms = [normalise_tm(tm) for tm in raw_tms]
    print(f"  {topo.num_nodes} nodes, {len(tms)} traffic matrices")

    max_deg = max(len(v) for v in topo.adjacency.values())
    dr_values = args.dr_values
    packet_counts = args.packet_counts

    scair_grid = np.zeros((len(packet_counts), len(dr_values)))
    ospf_grid  = np.zeros((len(packet_counts), len(dr_values)))

    total = len(packet_counts) * len(dr_values)
    done  = 0

    for pi, n_pkt in enumerate(packet_counts):
        for di, dr in enumerate(dr_values):
            done += 1
            print(f"[{done:2d}/{total}] D_r={dr:.1f}  packets={n_pkt:3d} ...", end="  ", flush=True)

            cfg = ScaIRConfig(
                feature_length=args.feature_length,
                neural_units=args.neural_units,
                packets_per_episode=n_pkt,
                distribution_ratio=dr,
                delay_input=args.delay_input,
                delay_init=args.delay_init,
                sigma_initial=0.1,
                sigma_min=0.1,
            )
            if topo.num_nodes > cfg.max_nodes:
                cfg.max_nodes = topo.num_nodes
            if max_deg > cfg.max_degree:
                cfg.max_degree = max_deg

            agents = build_agents(topo, cfg)
            if args.checkpoint:
                load_checkpoint(agents, args.checkpoint)

            s = eval_scair(topo, cfg, tms, agents, args.episodes, n_pkt)
            o = eval_ospf(topo, cfg, tms, args.episodes, n_pkt)

            scair_grid[pi, di] = s
            ospf_grid[pi, di]  = o
            winner = "ScaIR" if s < o else "OSPF "
            print(f"ScaIR={s:7.2f} ms  OSPF={o:7.2f} ms  diff={s-o:+6.2f} ms  [{winner} wins]")

    import os
    os.makedirs(os.path.dirname(args.save_plot) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Plot 1: line plots, one subplot per packet count
    # ------------------------------------------------------------------
    n_cols = len(packet_counts)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), sharey=False)
    if n_cols == 1:
        axes = [axes]

    for pi, (ax, n_pkt) in enumerate(zip(axes, packet_counts)):
        ax.plot(dr_values, scair_grid[pi], "o-",  label="ScaIR", color="tab:blue",   linewidth=2, markersize=6)
        ax.plot(dr_values, ospf_grid[pi],  "s--", label="OSPF",  color="tab:orange", linewidth=2, markersize=6)

        # Shade regions where each method wins
        ax.fill_between(dr_values, scair_grid[pi], ospf_grid[pi],
                        where=scair_grid[pi] <= ospf_grid[pi],
                        alpha=0.15, color="green", label="ScaIR wins")
        ax.fill_between(dr_values, scair_grid[pi], ospf_grid[pi],
                        where=scair_grid[pi] > ospf_grid[pi],
                        alpha=0.15, color="red",   label="OSPF wins")

        ax.set_title(f"{n_pkt} packets / episode", fontsize=11)
        ax.set_xlabel("Distribution ratio D_r")
        if pi == 0:
            ax.set_ylabel("Avg delivery time (ms)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(dr_values)

    fig.suptitle("ScaIR vs OSPF — avg delivery time across traffic conditions", fontsize=13)
    plt.tight_layout()
    plt.savefig(args.save_plot, dpi=150, bbox_inches="tight")
    print(f"\nLine plot saved  → {args.save_plot}")

    # ------------------------------------------------------------------
    # Plot 2: heatmap of (ScaIR - OSPF) difference
    # ------------------------------------------------------------------
    diff = scair_grid - ospf_grid   # negative = ScaIR wins, positive = OSPF wins
    abs_max = np.abs(diff).max() or 1.0

    heatmap_path = args.save_plot.replace(".png", "_heatmap.png")
    fig2, ax2 = plt.subplots(figsize=(max(6, 1.4 * len(dr_values)), max(3, 1.2 * len(packet_counts))))

    im = ax2.imshow(diff, cmap="RdYlGn_r",
                    vmin=-abs_max, vmax=abs_max, aspect="auto")
    plt.colorbar(im, ax=ax2, label="ScaIR − OSPF (ms)\n← ScaIR better   OSPF better →")

    ax2.set_xticks(range(len(dr_values)))
    ax2.set_xticklabels([f"{d:.1f}" for d in dr_values])
    ax2.set_yticks(range(len(packet_counts)))
    ax2.set_yticklabels([str(p) for p in packet_counts])
    ax2.set_xlabel("Distribution ratio D_r")
    ax2.set_ylabel("Packets per episode")
    ax2.set_title("ScaIR − OSPF delivery time (ms)\nGreen = ScaIR wins  |  Red = OSPF wins")

    for pi in range(len(packet_counts)):
        for di in range(len(dr_values)):
            ax2.text(di, pi, f"{diff[pi, di]:+.1f}",
                     ha="center", va="center", fontsize=9,
                     color="black" if abs(diff[pi, di]) < 0.6 * abs_max else "white")

    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved    → {heatmap_path}")


if __name__ == "__main__":
    main()
