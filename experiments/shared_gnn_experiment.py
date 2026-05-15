#!/usr/bin/env python3
"""
Shared SubGNN experiment for ScaIR (scair package).

Compares two ScaIR configurations trained from scratch on Abilene:
  A) Per-node SubGNN  -- each router has its own independent SubGNN
  B) Shared SubGNN    -- all routers share a single SubGNN (same f_w, g_w weights)

For each configuration we train for TRAIN_EPISODES episodes, then evaluate
for EVAL_EPISODES episodes with training disabled.  Results are compared on:
  - Training convergence (avg delivery time vs episode)
  - Final evaluation: avg delivery time, avg hops, packet delivery rate

Outputs
-------
  results/shared_gnn_training_curves.png  -- convergence comparison
  results/shared_gnn_eval_summary.png     -- bar chart of final metrics
  results/shared_gnn_experiment.json      -- raw numbers
"""

import heapq
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scair.agent import IRrAgent
from scair.config import ScaIRConfig
from scair.data_loader import load_all_traffic_matrices, load_topology, normalise_tm
from scair.environment import RoutingEnvironment
from scair.models import SubGNN
from train import build_agents, build_agents_shared_gnn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOPO_FILE  = "data/ABI/Topology.txt"
TM_DIR     = "data/ABI/TrafficMatrix"
RESULTS    = "results"
SEED       = 42

TRAIN_EPISODES = 500
EVAL_EPISODES  = 100
N_PACKETS      = 100
DIST_RATIO     = 0.7   # hot-spot fraction; higher → more congestion → ScaIR advantage
LOG_INTERVAL   = 50


# ---------------------------------------------------------------------------
# OSPF baseline (hop-count Dijkstra)
# ---------------------------------------------------------------------------

def ospf_avg_delivery_time(topo, tms, n_packets: int, dist_ratio: float, cfg: ScaIRConfig) -> dict:
    """Run OSPF (hop-count shortest path) and return delivery stats."""
    from scair.environment import Topology

    # Build shortest-path routing tables
    routes: Dict[int, Dict[int, int]] = {}
    for src in range(topo.num_nodes):
        dist = {n: float("inf") for n in range(topo.num_nodes)}
        prev: Dict[int, int] = {}
        dist[src] = 0
        pq = [(0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v in topo.adjacency[u]:
                nd = d + 1
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        next_hop: Dict[int, int] = {}
        for dst in range(topo.num_nodes):
            if dst == src or dist[dst] == float("inf"):
                continue
            node = dst
            while prev.get(node, src) != src:
                node = prev[node]
            next_hop[dst] = node
        routes[src] = next_hop

    env = RoutingEnvironment(topo, cfg)
    total_time, total_hops, delivered, total = 0.0, 0, 0, 0
    tm = tms[0]
    packets = env.generate_packets(tm, n_packets)

    for pkt in packets:
        src, dst = pkt.source, pkt.destination
        if src == dst:
            continue
        node, hops, t = src, 0, 0.0
        while node != dst and hops < cfg.max_hops:
            nh = routes[node].get(dst)
            if nh is None:
                break
            t += cfg.transmission_time
            node = nh
            hops += 1
        if node == dst:
            total_time += t
            total_hops += hops
            delivered += 1
        total += 1

    return {
        "avg_delivery_time": total_time / max(delivered, 1),
        "avg_hops": total_hops / max(delivered, 1),
        "delivery_rate": delivered / max(total, 1),
    }


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def run_training(
    label: str,
    topo,
    tms,
    cfg: ScaIRConfig,
    use_shared_gnn: bool,
) -> Tuple[List[dict], List[dict]]:
    """
    Train ScaIR for TRAIN_EPISODES episodes, then evaluate for EVAL_EPISODES.
    Returns (train_history, eval_history).
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if use_shared_gnn:
        agents = build_agents_shared_gnn(topo, cfg)
    else:
        agents = build_agents(topo, cfg)

    env = RoutingEnvironment(topo, cfg)
    train_history: List[dict] = []

    print(f"\n{'='*60}")
    print(f"  Training: {label}  ({TRAIN_EPISODES} episodes, {N_PACKETS} pkts/ep)")
    print(f"{'='*60}")
    t_start = time.time()

    for ep in range(1, TRAIN_EPISODES + 1):
        tm = tms[(ep - 1) % len(tms)]
        packets = env.generate_packets(tm, cfg.packets_per_episode)
        stats = env.run_episode(packets, agents, training=True)
        train_history.append(stats)

        # LR schedule (paper §5.1)
        if ep == 10:
            for ag in agents:
                ag.set_learning_rate(cfg.learning_rate)

        # Target network update
        if ep % cfg.target_update_freq == 0:
            for ag in agents:
                ag.update_target()

        # Sigma decay
        if ep % cfg.sigma_decay_freq == 0:
            for ag in agents:
                ag.decay_sigma()

        if ep % LOG_INTERVAL == 0 or ep == 1:
            elapsed = time.time() - t_start
            print(
                f"  [{label}] ep {ep:4d}/{TRAIN_EPISODES}  "
                f"avg_time={stats['avg_delivery_time']:7.3f} ms  "
                f"delivered={stats['delivered']:3d}/{cfg.packets_per_episode}  "
                f"loss={stats['avg_loss']:.4f}  "
                f"sigma={agents[0].sigma:.2f}  "
                f"({elapsed:.1f}s)"
            )

    print(f"\n  Evaluating {label} ({EVAL_EPISODES} eps)...")
    eval_history: List[dict] = []
    for ep in range(1, EVAL_EPISODES + 1):
        tm = tms[(TRAIN_EPISODES + ep - 1) % len(tms)]
        packets = env.generate_packets(tm, cfg.packets_per_episode)
        stats = env.run_episode(packets, agents, training=False)
        eval_history.append(stats)

    avg_time = np.mean([s["avg_delivery_time"] for s in eval_history])
    avg_hops = np.mean([s["avg_hops"] for s in eval_history])
    avg_rate = np.mean([s["delivered"] / max(cfg.packets_per_episode, 1) for s in eval_history])
    print(f"  {label} eval: time={avg_time:.3f} ms  hops={avg_hops:.2f}  rate={avg_rate:.3f}")

    return train_history, eval_history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def smooth(values: List[float], window: int = 20) -> List[float]:
    out = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out.append(float(np.mean(values[lo:hi])))
    return out


def plot_training_curves(
    per_node_hist: List[dict],
    shared_hist: List[dict],
    ospf_time: float,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Shared SubGNN vs Per-Node SubGNN: Training Convergence", fontsize=14)

    episodes = list(range(1, len(per_node_hist) + 1))

    # Delivery time
    ax = axes[0]
    pn_times = smooth([s["avg_delivery_time"] for s in per_node_hist])
    sh_times = smooth([s["avg_delivery_time"] for s in shared_hist])
    ax.plot(episodes, pn_times, label="Per-node SubGNN", color="steelblue", linewidth=1.5)
    ax.plot(episodes, sh_times, label="Shared SubGNN", color="darkorange", linewidth=1.5)
    ax.axhline(ospf_time, color="green", linestyle="--", linewidth=1.5, label=f"OSPF ({ospf_time:.2f} ms)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Delivery Time (ms)")
    ax.set_title("Avg Delivery Time")
    ax.legend()
    ax.grid(alpha=0.3)

    # Loss
    ax = axes[1]
    pn_loss = smooth([s.get("avg_loss", 0) for s in per_node_hist])
    sh_loss = smooth([s.get("avg_loss", 0) for s in shared_hist])
    ax.plot(episodes, pn_loss, label="Per-node SubGNN", color="steelblue", linewidth=1.5)
    ax.plot(episodes, sh_loss, label="Shared SubGNN", color="darkorange", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg MSE Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_eval_summary(
    per_node_eval: List[dict],
    shared_eval: List[dict],
    ospf_stats: dict,
    cfg: ScaIRConfig,
    out_path: str,
) -> None:
    labels = ["Per-node SubGNN", "Shared SubGNN", "OSPF"]
    colors = ["steelblue", "darkorange", "green"]

    def mean_stat(hist, key):
        return float(np.mean([s[key] for s in hist]))

    times = [
        mean_stat(per_node_eval, "avg_delivery_time"),
        mean_stat(shared_eval, "avg_delivery_time"),
        ospf_stats["avg_delivery_time"],
    ]
    hops = [
        mean_stat(per_node_eval, "avg_hops"),
        mean_stat(shared_eval, "avg_hops"),
        ospf_stats["avg_hops"],
    ]
    rates = [
        mean_stat(per_node_eval, "delivered") / cfg.packets_per_episode,
        mean_stat(shared_eval, "delivered") / cfg.packets_per_episode,
        ospf_stats["delivery_rate"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Shared SubGNN vs Per-Node SubGNN: Evaluation Summary", fontsize=14)

    for ax, values, title, ylabel in zip(
        axes,
        [times, hops, rates],
        ["Avg Delivery Time", "Avg Hops", "Delivery Rate"],
        ["ms", "hops", "fraction"],
    ):
        bars = ax.bar(labels, values, color=colors, alpha=0.8)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=10,
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)

    print("Loading topology and traffic matrices...")
    topo = load_topology(TOPO_FILE)
    raw_tms = load_all_traffic_matrices(TM_DIR, topo.num_nodes)
    tms = [normalise_tm(tm) for tm in raw_tms]
    print(f"  {topo.num_nodes} nodes, {len(tms)} TMs")

    # Shared config for both runs
    cfg = ScaIRConfig()
    cfg.packets_per_episode = N_PACKETS
    cfg.distribution_ratio  = DIST_RATIO
    cfg.max_episodes        = TRAIN_EPISODES

    # Adjust bounds for topology
    max_deg = max(len(v) for v in topo.adjacency.values())
    if topo.num_nodes > cfg.max_nodes:
        cfg.max_nodes = topo.num_nodes
    if max_deg > cfg.max_degree:
        cfg.max_degree = max_deg

    # OSPF baseline
    print("\nRunning OSPF baseline...")
    ospf_stats = ospf_avg_delivery_time(topo, tms, N_PACKETS, DIST_RATIO, cfg)
    print(f"  OSPF: time={ospf_stats['avg_delivery_time']:.3f} ms  "
          f"hops={ospf_stats['avg_hops']:.2f}  "
          f"rate={ospf_stats['delivery_rate']:.3f}")

    # Per-node SubGNN run
    pn_train, pn_eval = run_training(
        label="Per-node SubGNN",
        topo=topo, tms=tms, cfg=cfg,
        use_shared_gnn=False,
    )

    # Shared SubGNN run
    sh_train, sh_eval = run_training(
        label="Shared SubGNN",
        topo=topo, tms=tms, cfg=cfg,
        use_shared_gnn=True,
    )

    # Plots
    plot_training_curves(
        pn_train, sh_train, ospf_stats["avg_delivery_time"],
        os.path.join(RESULTS, "shared_gnn_training_curves.png"),
    )
    plot_eval_summary(
        pn_eval, sh_eval, ospf_stats, cfg,
        os.path.join(RESULTS, "shared_gnn_eval_summary.png"),
    )

    # JSON
    def summarise(hist, cfg):
        times = [s["avg_delivery_time"] for s in hist]
        return {
            "avg_delivery_time": float(np.mean(times)),
            "std_delivery_time": float(np.std(times)),
            "avg_hops": float(np.mean([s["avg_hops"] for s in hist])),
            "delivery_rate": float(np.mean([s["delivered"] for s in hist])) / cfg.packets_per_episode,
        }

    results = {
        "config": {
            "train_episodes": TRAIN_EPISODES,
            "eval_episodes": EVAL_EPISODES,
            "n_packets": N_PACKETS,
            "dist_ratio": DIST_RATIO,
            "seed": SEED,
        },
        "ospf": ospf_stats,
        "per_node_gnn": {
            "train": summarise(pn_train, cfg),
            "eval":  summarise(pn_eval, cfg),
            "train_curve": [s["avg_delivery_time"] for s in pn_train],
        },
        "shared_gnn": {
            "train": summarise(sh_train, cfg),
            "eval":  summarise(sh_eval, cfg),
            "train_curve": [s["avg_delivery_time"] for s in sh_train],
        },
    }
    json_path = os.path.join(RESULTS, "shared_gnn_experiment.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # Summary table
    print("\n" + "="*60)
    print("FINAL EVALUATION SUMMARY")
    print("="*60)
    for name, stats in [("Per-node SubGNN", results["per_node_gnn"]["eval"]),
                         ("Shared SubGNN",   results["shared_gnn"]["eval"]),
                         ("OSPF",            results["ospf"])]:
        print(f"  {name:<22}  time={stats['avg_delivery_time']:7.3f} ms  "
              f"hops={stats['avg_hops']:5.2f}  rate={stats['delivery_rate']:.3f}")
    print("="*60)


if __name__ == "__main__":
    main()
