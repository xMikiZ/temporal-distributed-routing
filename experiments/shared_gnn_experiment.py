#!/usr/bin/env python3
"""
Shared SubGNN experiment for ScaIR (scair package).

Compares two ScaIR configurations trained from scratch on Abilene:
  A) Per-node SubGNN  -- each router has its own independent SubGNN
  B) Shared SubGNN    -- all routers share a single SubGNN (same f_w, g_w weights)

Sweeps over multiple hot-spot ratios (D_r) to show how congestion level
affects the relative advantage of each approach versus OSPF.

OSPF uses a heap-based queue simulation (same cost model as ScaIR: queuing
time + 1ms transmission), giving a fair apples-to-apples comparison.

Outputs
-------
  results/shared_gnn_dr_sweep.png         -- delivery time vs D_r (main result)
  results/shared_gnn_training_curves.png  -- convergence at each D_r
  results/shared_gnn_experiment.json      -- raw numbers
"""

import heapq
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scair.config import ScaIRConfig
from scair.data_loader import load_all_traffic_matrices, load_topology, normalise_tm
from scair.environment import RoutingEnvironment
from train import build_agents, build_agents_shared_gnn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOPO_FILE      = "data/ABI/Topology.txt"
TM_DIR         = "data/ABI/TrafficMatrix"
RESULTS        = "results"
SEED           = 42

DR_VALUES      = [0.0, 0.2, 0.4, 0.6, 0.8]   # hot-spot fractions to sweep
TRAIN_EPISODES = 300                            # per (D_r, config) pair
EVAL_EPISODES  = 50
N_PACKETS      = 100
LOG_INTERVAL   = 50


# ---------------------------------------------------------------------------
# OSPF baseline — heap-based with queue simulation (same cost as ScaIR)
# ---------------------------------------------------------------------------

def _dijkstra_next_hop(topo, source: int) -> Dict[int, int]:
    dist = {n: float("inf") for n in range(topo.num_nodes)}
    prev: Dict[int, int] = {}
    dist[source] = 0
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v in topo.adjacency[u]:
            nd = d + 1
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    next_hop: Dict[int, int] = {}
    for dst in range(topo.num_nodes):
        if dst == source or dist[dst] == float("inf"):
            continue
        node = dst
        while prev.get(node, source) != source:
            node = prev[node]
        next_hop[dst] = node
    return next_hop


def run_ospf_episodes(topo, cfg: ScaIRConfig, tms, n_episodes: int, n_packets: int) -> dict:
    """Run OSPF for n_episodes episodes; returns mean delivery time, hops, rate."""
    routes = {n: _dijkstra_next_hop(topo, n) for n in range(topo.num_nodes)}
    env = RoutingEnvironment(topo, cfg)

    all_times, all_hops, all_rates = [], [], []
    for ep in range(n_episodes):
        tm = tms[ep % len(tms)]
        packets = env.generate_packets(tm, n_packets)

        heap: list = []
        for pkt in packets:
            heapq.heappush(heap, (pkt.birth_time, pkt.pid, pkt, pkt.source, None))

        delivery_times: List[float] = []
        hop_counts: Dict[int, int] = defaultdict(int)
        queue_len: Dict[Tuple[int, int], int] = defaultdict(int)

        while heap:
            t, _, pkt, node, prev_node = heapq.heappop(heap)
            if prev_node is not None:
                queue_len[(prev_node, node)] = max(0, queue_len[(prev_node, node)] - 1)
            if node == pkt.destination:
                delivery_times.append(t - pkt.birth_time)
                continue
            if hop_counts[pkt.pid] >= cfg.max_hops:
                continue
            next_node = routes.get(node, {}).get(pkt.destination)
            if next_node is None:
                continue
            q = queue_len[(node, next_node)]
            cost = q * cfg.queue_time_per_packet + cfg.transmission_time
            queue_len[(node, next_node)] += 1
            hop_counts[pkt.pid] += 1
            heapq.heappush(heap, (t + cost, pkt.pid, pkt, next_node, node))

        n_delivered = len(delivery_times)
        all_times.append(float(np.mean(delivery_times)) if delivery_times else float("inf"))
        all_hops.append(float(np.mean(list(hop_counts.values()))) if hop_counts else 0.0)
        all_rates.append(n_delivered / max(n_packets, 1))

    return {
        "avg_delivery_time": float(np.mean(all_times)),
        "avg_hops": float(np.mean(all_hops)),
        "delivery_rate": float(np.mean(all_rates)),
    }


# ---------------------------------------------------------------------------
# Training + evaluation for one (D_r, config) pair
# ---------------------------------------------------------------------------

def run_one(label: str, topo, tms, cfg: ScaIRConfig, use_shared_gnn: bool) -> dict:
    """Train ScaIR for TRAIN_EPISODES, evaluate for EVAL_EPISODES."""
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    agents = build_agents_shared_gnn(topo, cfg) if use_shared_gnn else build_agents(topo, cfg)
    env    = RoutingEnvironment(topo, cfg)

    train_curve: List[float] = []
    t_start = time.time()

    for ep in range(1, TRAIN_EPISODES + 1):
        tm    = tms[(ep - 1) % len(tms)]
        pkts  = env.generate_packets(tm, cfg.packets_per_episode)
        stats = env.run_episode(pkts, agents, training=True)
        train_curve.append(stats["avg_delivery_time"])

        if ep == 10:
            for ag in agents: ag.set_learning_rate(cfg.learning_rate)
        if ep % cfg.target_update_freq == 0:
            for ag in agents: ag.update_target()
        if ep % cfg.sigma_decay_freq == 0:
            for ag in agents: ag.decay_sigma()

        if ep % LOG_INTERVAL == 0 or ep == 1:
            elapsed = time.time() - t_start
            print(f"  [{label}] ep {ep:3d}/{TRAIN_EPISODES}  "
                  f"time={stats['avg_delivery_time']:6.2f}ms  "
                  f"sigma={agents[0].sigma:.2f}  ({elapsed:.0f}s)")

    # Evaluate
    eval_times: List[float] = []
    for ep in range(1, EVAL_EPISODES + 1):
        tm    = tms[(TRAIN_EPISODES + ep - 1) % len(tms)]
        pkts  = env.generate_packets(tm, cfg.packets_per_episode)
        stats = env.run_episode(pkts, agents, training=False)
        eval_times.append(stats["avg_delivery_time"])

    return {
        "avg_delivery_time": float(np.mean(eval_times)),
        "std_delivery_time": float(np.std(eval_times)),
        "train_curve": train_curve,
    }


# ---------------------------------------------------------------------------
# Smooth helper
# ---------------------------------------------------------------------------

def smooth(values: List[float], w: int = 15) -> List[float]:
    out = []
    for i in range(len(values)):
        lo = max(0, i - w // 2)
        hi = min(len(values), i + w // 2 + 1)
        out.append(float(np.mean(values[lo:hi])))
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_dr_sweep(results: dict, out_path: str) -> None:
    drs   = sorted(results.keys())
    ospf  = [results[d]["ospf"]["avg_delivery_time"] for d in drs]
    pn    = [results[d]["per_node"]["avg_delivery_time"] for d in drs]
    sh    = [results[d]["shared"]["avg_delivery_time"] for d in drs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Shared SubGNN vs Per-Node SubGNN: D_r Sweep (Abilene, 300 eps training)", fontsize=13)

    # Left: absolute delivery time
    ax = axes[0]
    ax.plot(drs, ospf, "o--", color="green",      label="OSPF",           linewidth=2)
    ax.plot(drs, pn,   "s-",  color="steelblue",  label="Per-node SubGNN", linewidth=2)
    ax.plot(drs, sh,   "^-",  color="darkorange",  label="Shared SubGNN",   linewidth=2)
    ax.set_xlabel("Hot-spot ratio D_r")
    ax.set_ylabel("Avg Delivery Time (ms)")
    ax.set_title("Delivery Time vs D_r")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xticks(drs)

    # Right: ScaIR gain over OSPF  (positive = ScaIR faster)
    ax = axes[1]
    gain_pn = [(o - p) / max(o, 1e-6) * 100 for o, p in zip(ospf, pn)]
    gain_sh = [(o - s) / max(o, 1e-6) * 100 for o, s in zip(ospf, sh)]
    ax.plot(drs, gain_pn, "s-",  color="steelblue",  label="Per-node SubGNN", linewidth=2)
    ax.plot(drs, gain_sh, "^-",  color="darkorange",  label="Shared SubGNN",   linewidth=2)
    ax.axhline(0, color="green", linestyle="--", linewidth=1.5, label="OSPF parity")
    ax.set_xlabel("Hot-spot ratio D_r")
    ax.set_ylabel("Improvement over OSPF (%)")
    ax.set_title("ScaIR gain vs OSPF")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xticks(drs)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_training_curves(results: dict, out_path: str) -> None:
    drs = sorted(results.keys())
    n   = len(drs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    fig.suptitle("Training Convergence by D_r", fontsize=13)
    if n == 1:
        axes = [axes]

    cmap = plt.cm.plasma
    for i, (dr, ax) in enumerate(zip(drs, axes)):
        color_pn = "steelblue"
        color_sh = "darkorange"
        pn_curve = smooth(results[dr]["per_node"]["train_curve"])
        sh_curve = smooth(results[dr]["shared"]["train_curve"])
        eps = list(range(1, len(pn_curve) + 1))
        ax.plot(eps, pn_curve, color=color_pn, linewidth=1.5, label="Per-node")
        ax.plot(eps, sh_curve, color=color_sh, linewidth=1.5, label="Shared")
        ax.axhline(results[dr]["ospf"]["avg_delivery_time"], color="green",
                   linestyle="--", linewidth=1.2, label="OSPF")
        ax.set_title(f"D_r = {dr}")
        ax.set_xlabel("Episode")
        if i == 0:
            ax.set_ylabel("Avg Delivery Time (ms)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

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
    topo    = load_topology(TOPO_FILE)
    raw_tms = load_all_traffic_matrices(TM_DIR, topo.num_nodes)
    tms     = [normalise_tm(tm) for tm in raw_tms]
    print(f"  {topo.num_nodes} nodes, {len(tms)} TMs")

    # Fix topology bounds once
    base_cfg = ScaIRConfig()
    max_deg  = max(len(v) for v in topo.adjacency.values())
    if topo.num_nodes > base_cfg.max_nodes:  base_cfg.max_nodes  = topo.num_nodes
    if max_deg        > base_cfg.max_degree: base_cfg.max_degree = max_deg

    all_results: dict = {}

    for dr in DR_VALUES:
        print(f"\n{'='*60}")
        print(f"  D_r = {dr}  ({TRAIN_EPISODES} training eps, {N_PACKETS} pkts/ep)")
        print(f"{'='*60}")

        import copy
        cfg = copy.copy(base_cfg)
        cfg.distribution_ratio  = dr
        cfg.packets_per_episode = N_PACKETS

        # OSPF baseline (queue-aware)
        print(f"  OSPF baseline ({EVAL_EPISODES} eps)...")
        ospf = run_ospf_episodes(topo, cfg, tms, EVAL_EPISODES, N_PACKETS)
        print(f"    OSPF: time={ospf['avg_delivery_time']:.3f} ms  "
              f"hops={ospf['avg_hops']:.2f}  rate={ospf['delivery_rate']:.3f}")

        # Per-node SubGNN
        print(f"  Training Per-node SubGNN...")
        pn = run_one(f"D_r={dr} per-node", topo, tms, cfg, use_shared_gnn=False)
        print(f"    Per-node eval: {pn['avg_delivery_time']:.3f} ms")

        # Shared SubGNN
        print(f"  Training Shared SubGNN...")
        sh = run_one(f"D_r={dr} shared", topo, tms, cfg, use_shared_gnn=True)
        print(f"    Shared eval:   {sh['avg_delivery_time']:.3f} ms")

        all_results[dr] = {"ospf": ospf, "per_node": pn, "shared": sh}

    # Plots
    print("\nGenerating plots...")
    plot_dr_sweep(all_results,
                  os.path.join(RESULTS, "shared_gnn_dr_sweep.png"))
    plot_training_curves(all_results,
                         os.path.join(RESULTS, "shared_gnn_training_curves.png"))

    # JSON (drop raw train curves to keep file small — keep summary only)
    json_out = {}
    for dr, res in all_results.items():
        json_out[str(dr)] = {
            "ospf":     res["ospf"],
            "per_node": {k: v for k, v in res["per_node"].items() if k != "train_curve"},
            "shared":   {k: v for k, v in res["shared"].items()   if k != "train_curve"},
        }
    json_path = os.path.join(RESULTS, "shared_gnn_experiment.json")
    with open(json_path, "w") as f:
        json.dump({"config": {"dr_values": DR_VALUES, "train_episodes": TRAIN_EPISODES,
                               "eval_episodes": EVAL_EPISODES, "n_packets": N_PACKETS,
                               "seed": SEED},
                   "results": json_out}, f, indent=2)
    print(f"  Saved: {json_path}")

    # Summary table
    print("\n" + "="*70)
    print(f"{'D_r':>6}  {'OSPF':>8}  {'Per-node':>10}  {'Shared':>10}  {'Gain(PN)':>9}  {'Gain(Sh)':>9}")
    print("="*70)
    for dr in DR_VALUES:
        r = all_results[dr]
        o  = r["ospf"]["avg_delivery_time"]
        pn = r["per_node"]["avg_delivery_time"]
        sh = r["shared"]["avg_delivery_time"]
        def gain(x): return (o - x) / max(o, 1e-6) * 100
        print(f"  {dr:.1f}   {o:8.3f}  {pn:10.3f}  {sh:10.3f}  {gain(pn):+8.1f}%  {gain(sh):+8.1f}%")
    print("="*70)


if __name__ == "__main__":
    main()
