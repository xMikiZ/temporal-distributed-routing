#!/usr/bin/env python3
"""
Ablation: replace the SubGNN feature vector with a fixed encoding.

Compares four no-GNN variants against OSPF (and optionally a reference
ScaIR result loaded from an existing comparison JSON):

  A) OneHot per-node Q   -- V_n = one-hot(node_id),   independent Q-net per node
  B) OneHot shared Q     -- V_n = one-hot(node_id),   one shared Q-net for all nodes
  C) NeighborMask per-node Q -- V_n = binary neighbour mask, independent Q-nets
  D) NeighborMask shared Q   -- V_n = binary neighbour mask, one shared Q-net

Goal: determine whether the learned GNN feature vector provides information
beyond simple topology encodings (one-hot identity vs neighbour adjacency).

Usage
-----
  python experiments/ablation_no_gnn.py \\
      --topo data/ABI/Topology.txt \\
      --tm_dir data/ABI/TrafficMatrix \\
      --results results/05_no_gnn_abi \\
      --reference_json results/01_dr_comparison/comparison_experiment.json \\
      --episodes 300 --eval_episodes 50 --action_method ucb

Outputs
-------
  <results>/ablation_dr_sweep.png
  <results>/ablation_training_curves.png
  <results>/ablation_no_gnn.json
"""

import argparse
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
from scair.models import OneHotSubGNN, NeighborMaskSubGNN
from train import build_agents_no_gnn, build_agents_no_gnn_shared_q


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="No-GNN ablation — D_r sweep")
    p.add_argument("--topo",           default="data/ABI/Topology.txt")
    p.add_argument("--tm_dir",         default="data/ABI/TrafficMatrix")
    p.add_argument("--results",        default="results/05_no_gnn_abi")
    p.add_argument("--reference_json", default=None,
                   help="Path to existing comparison_experiment.json to overlay "
                        "per-node ScaIR results on plots (optional)")
    p.add_argument("--episodes",       type=int,   default=300)
    p.add_argument("--eval_episodes",  type=int,   default=50)
    p.add_argument("--packets",        type=int,   default=100)
    p.add_argument("--dr_values",      type=float, nargs="+",
                   default=[0.0, 0.2, 0.4, 0.6, 0.8])
    p.add_argument("--action_method",  default="ucb",
                   choices=["epsilon_greedy", "ucb"])
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--log_interval",   type=int,   default=50)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Globals (overwritten from args in main)
# ---------------------------------------------------------------------------

TOPO_FILE      = "data/ABI/Topology.txt"
TM_DIR         = "data/ABI/TrafficMatrix"
RESULTS        = "results/05_no_gnn_abi"
SEED           = 42
DR_VALUES      = [0.0, 0.2, 0.4, 0.6, 0.8]
TRAIN_EPISODES = 300
EVAL_EPISODES  = 50
N_PACKETS      = 100
LOG_INTERVAL   = 50
ACTION_METHOD  = "ucb"

VARIANTS = [
    ("onehot_pernode",   "OneHot per-node Q",   "steelblue",    "o"),
    ("onehot_sharedq",   "OneHot shared Q",     "dodgerblue",   "s"),
    ("nbrmask_pernode",  "NbrMask per-node Q",  "darkorange",   "^"),
    ("nbrmask_sharedq",  "NbrMask shared Q",    "coral",        "D"),
]


# ---------------------------------------------------------------------------
# OSPF baseline (identical to comparison_experiment.py)
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


def run_ospf(topo, cfg: ScaIRConfig, tms, n_episodes: int, n_packets: int) -> dict:
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
        all_times.append(float(np.mean(delivery_times)) if delivery_times else float("inf"))
        all_hops.append(float(np.mean(list(hop_counts.values()))) if hop_counts else 0.0)
        all_rates.append(len(delivery_times) / max(n_packets, 1))
    return {
        "avg_delivery_time": float(np.mean(all_times)),
        "avg_hops":          float(np.mean(all_hops)),
        "delivery_rate":     float(np.mean(all_rates)),
    }


# ---------------------------------------------------------------------------
# Training + evaluation for one (D_r, variant) pair
# ---------------------------------------------------------------------------

def _build(variant_key: str, topo, cfg: ScaIRConfig) -> list:
    if variant_key == "onehot_pernode":
        return build_agents_no_gnn(topo, cfg, OneHotSubGNN)
    if variant_key == "onehot_sharedq":
        return build_agents_no_gnn_shared_q(topo, cfg, OneHotSubGNN)
    if variant_key == "nbrmask_pernode":
        return build_agents_no_gnn(topo, cfg, NeighborMaskSubGNN)
    if variant_key == "nbrmask_sharedq":
        return build_agents_no_gnn_shared_q(topo, cfg, NeighborMaskSubGNN)
    raise ValueError(f"Unknown variant: {variant_key}")


def run_variant(label: str, variant_key: str, topo, tms, cfg: ScaIRConfig) -> dict:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    agents = _build(variant_key, topo, cfg)
    env = RoutingEnvironment(topo, cfg)
    train_curve: List[float] = []
    t_start = time.time()

    for ep in range(1, TRAIN_EPISODES + 1):
        tm   = tms[(ep - 1) % len(tms)]
        pkts = env.generate_packets(tm, cfg.packets_per_episode)
        stats = env.run_episode(pkts, agents, training=True)
        train_curve.append(stats["avg_delivery_time"])

        if ep == 10:
            for ag in agents:
                ag.set_learning_rate(cfg.learning_rate)
        if ep % cfg.target_update_freq == 0:
            for ag in agents:
                ag.update_target()
        if ep % cfg.sigma_decay_freq == 0:
            for ag in agents:
                ag.decay_sigma()

        if ep % LOG_INTERVAL == 0 or ep == 1:
            elapsed = time.time() - t_start
            print(f"  [{label}] ep {ep:3d}/{TRAIN_EPISODES}  "
                  f"time={stats['avg_delivery_time']:6.2f}ms  "
                  f"sigma={agents[0].sigma:.2f}  ({elapsed:.0f}s)")

    eval_times: List[float] = []
    for ep in range(1, EVAL_EPISODES + 1):
        tm   = tms[(TRAIN_EPISODES + ep - 1) % len(tms)]
        pkts = env.generate_packets(tm, cfg.packets_per_episode)
        stats = env.run_episode(pkts, agents, training=False)
        eval_times.append(stats["avg_delivery_time"])

    return {
        "avg_delivery_time": float(np.mean(eval_times)),
        "std_delivery_time": float(np.std(eval_times)),
        "train_curve":       train_curve,
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

def plot_dr_sweep(results: dict, ref_results: dict, out_path: str) -> None:
    drs  = sorted(results.keys())
    ospf = [results[d]["ospf"]["avg_delivery_time"] for d in drs]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("No-GNN Ablation vs OSPF: D_r Sweep", fontsize=13)

    ax = axes[0]
    ax.plot(drs, ospf, "o--", color="green", label="OSPF", linewidth=2)

    # Reference: per-node ScaIR from existing experiment (if provided)
    if ref_results:
        ref_times = []
        for d in drs:
            k = str(d) if str(d) in ref_results else f"{d:.1f}"
            ref_times.append(ref_results.get(k, {}).get("per_node", {}).get("avg_delivery_time", None))
        if all(v is not None for v in ref_times):
            ax.plot(drs, ref_times, "x--", color="black", linewidth=2,
                    label="ScaIR per-node (GNN)")

    for key, label, color, marker in VARIANTS:
        times = [results[d][key]["avg_delivery_time"] for d in drs]
        ax.plot(drs, times, f"{marker}-", color=color, label=label, linewidth=2)
    ax.set_xlabel("Hot-spot ratio D_r"); ax.set_ylabel("Avg Delivery Time (ms)")
    ax.set_title("Delivery Time vs D_r"); ax.legend(); ax.grid(alpha=0.3); ax.set_xticks(drs)

    ax = axes[1]
    # Reference gain
    if ref_results:
        ref_gains = []
        for o, d in zip(ospf, drs):
            k = str(d) if str(d) in ref_results else f"{d:.1f}"
            v = ref_results.get(k, {}).get("per_node", {}).get("avg_delivery_time", None)
            ref_gains.append((o - v) / max(o, 1e-6) * 100 if v else None)
        if all(v is not None for v in ref_gains):
            ax.plot(drs, ref_gains, "x--", color="black", linewidth=2,
                    label="ScaIR per-node (GNN)")

    for key, label, color, marker in VARIANTS:
        gains = [(o - results[d][key]["avg_delivery_time"]) / max(o, 1e-6) * 100
                 for o, d in zip(ospf, drs)]
        ax.plot(drs, gains, f"{marker}-", color=color, label=label, linewidth=2)
    ax.axhline(0, color="green", linestyle="--", linewidth=1.5, label="OSPF parity")
    ax.set_xlabel("Hot-spot ratio D_r"); ax.set_ylabel("Improvement over OSPF (%)")
    ax.set_title("Gain vs OSPF"); ax.legend(); ax.grid(alpha=0.3); ax.set_xticks(drs)

    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_training_curves(results: dict, out_path: str) -> None:
    drs = sorted(results.keys())
    n   = len(drs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    fig.suptitle("No-GNN Ablation: Training Convergence by D_r", fontsize=13)
    if n == 1:
        axes = [axes]

    for i, (dr, ax) in enumerate(zip(drs, axes)):
        for key, label, color, _ in VARIANTS:
            curve = smooth(results[dr][key]["train_curve"])
            ax.plot(range(1, len(curve) + 1), curve, color=color, linewidth=1.5, label=label)
        ax.axhline(results[dr]["ospf"]["avg_delivery_time"], color="green",
                   linestyle="--", linewidth=1.2, label="OSPF")
        ax.set_title(f"D_r = {dr}"); ax.set_xlabel("Episode")
        if i == 0:
            ax.set_ylabel("Avg Delivery Time (ms)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global TOPO_FILE, TM_DIR, RESULTS, SEED, DR_VALUES, TRAIN_EPISODES, \
           EVAL_EPISODES, N_PACKETS, LOG_INTERVAL, ACTION_METHOD

    args = parse_args()
    TOPO_FILE      = args.topo
    TM_DIR         = args.tm_dir
    RESULTS        = args.results
    SEED           = args.seed
    DR_VALUES      = args.dr_values
    TRAIN_EPISODES = args.episodes
    EVAL_EPISODES  = args.eval_episodes
    N_PACKETS      = args.packets
    LOG_INTERVAL   = args.log_interval
    ACTION_METHOD  = args.action_method

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(RESULTS, exist_ok=True)

    # Load optional reference results (per-node ScaIR with GNN)
    ref_results = {}
    if args.reference_json and os.path.exists(args.reference_json):
        with open(args.reference_json) as f:
            ref_data = json.load(f)
        ref_results = ref_data.get("results", {})
        print(f"Loaded reference results from {args.reference_json}")

    print("Loading topology and traffic matrices...")
    topo    = load_topology(TOPO_FILE)
    raw_tms = load_all_traffic_matrices(TM_DIR, topo.num_nodes)
    tms     = [normalise_tm(tm) for tm in raw_tms]
    print(f"  {topo.num_nodes} nodes, {len(tms)} TMs")

    base_cfg = ScaIRConfig()
    max_deg  = max(len(v) for v in topo.adjacency.values())
    if topo.num_nodes > base_cfg.max_nodes:  base_cfg.max_nodes  = topo.num_nodes
    if max_deg        > base_cfg.max_degree: base_cfg.max_degree = max_deg
    base_cfg.action_method = ACTION_METHOD

    all_results: dict = {}

    for dr in DR_VALUES:
        import copy
        cfg = copy.copy(base_cfg)
        cfg.distribution_ratio  = dr
        cfg.packets_per_episode = N_PACKETS

        print(f"\n{'='*65}")
        print(f"  D_r = {dr}  ({TRAIN_EPISODES} training eps, {N_PACKETS} pkts/ep)")
        print(f"{'='*65}")

        print(f"  OSPF baseline ({EVAL_EPISODES} eps)...")
        ospf = run_ospf(topo, cfg, tms, EVAL_EPISODES, N_PACKETS)
        print(f"    OSPF: {ospf['avg_delivery_time']:.3f} ms  hops={ospf['avg_hops']:.2f}")

        dr_results: dict = {"ospf": ospf}
        for key, label, _, _ in VARIANTS:
            print(f"  Training {label}...")
            res = run_variant(f"D_r={dr} {label}", key, topo, tms, cfg)
            dr_results[key] = res
            o    = ospf["avg_delivery_time"]
            gain = (o - res["avg_delivery_time"]) / max(o, 1e-6) * 100
            print(f"    {label}: {res['avg_delivery_time']:.3f} ms  ({gain:+.1f}% vs OSPF)")

        all_results[dr] = dr_results

    print("\nGenerating plots...")
    plot_dr_sweep(all_results, ref_results,
                  os.path.join(RESULTS, "ablation_dr_sweep.png"))
    plot_training_curves(all_results,
                         os.path.join(RESULTS, "ablation_training_curves.png"))

    # Save JSON (drop train_curves)
    json_out = {}
    for dr, res in all_results.items():
        json_out[str(dr)] = {
            "ospf": res["ospf"],
            **{k: {x: v for x, v in res[k].items() if x != "train_curve"}
               for k, _, _, _ in VARIANTS},
        }
    json_path = os.path.join(RESULTS, "ablation_no_gnn.json")
    with open(json_path, "w") as f:
        json.dump({"config": {"dr_values": DR_VALUES, "train_episodes": TRAIN_EPISODES,
                               "eval_episodes": EVAL_EPISODES, "n_packets": N_PACKETS,
                               "seed": SEED, "action_method": ACTION_METHOD},
                   "results": json_out}, f, indent=2)
    print(f"  Saved: {json_path}")

    # Summary table
    print("\n" + "="*90)
    print(f"{'D_r':>5}  {'OSPF':>8}" + "".join(f"  {k[0]:>18}  {'gain':>7}" for k in VARIANTS))
    print("="*90)
    for dr in DR_VALUES:
        r = all_results[dr]
        o = r["ospf"]["avg_delivery_time"]
        row = f"  {dr:.1f}   {o:8.3f}"
        for key, *_ in VARIANTS:
            t = r[key]["avg_delivery_time"]
            g = (o - t) / max(o, 1e-6) * 100
            row += f"  {t:18.3f}  {g:+6.1f}%"
        print(row)
    print("="*90)


if __name__ == "__main__":
    main()
