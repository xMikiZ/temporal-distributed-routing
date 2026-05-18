#!/usr/bin/env python3
"""
Full comparison experiment: SubGNN variants vs OSPF across D_r values.

Variants compared:
  A) Per-node SubGNN  -- independent SubGNN per router (mean aggregation)
  B) Shared SubGNN    -- all routers share one SubGNN (mean aggregation)
  C) Per-node Attention -- independent SubGNN per router (dot-product attention)
  D) Shared Attention -- all routers share one attention SubGNN

Sweeps D_r in [0.0, 0.2, 0.4, 0.6, 0.8] against a queue-simulating OSPF baseline.

Outputs
-------
  results/comparison_dr_sweep.png       -- delivery time vs D_r (all variants)
  results/comparison_gain.png           -- % gain over OSPF vs D_r
  results/comparison_experiment.json    -- raw numbers
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
from scair.models import AttentionSubGNN
from train import build_agents, build_agents_shared_gnn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOPO_FILE      = "data/ABI/Topology.txt"
TM_DIR         = "data/ABI/TrafficMatrix"
RESULTS        = "results"
SEED           = 42

DR_VALUES      = [0.0, 0.2, 0.4, 0.6, 0.8]
TRAIN_EPISODES = 300
EVAL_EPISODES  = 50
N_PACKETS      = 100
LOG_INTERVAL   = 100

VARIANTS = [
    ("per_node",        "Per-node SubGNN",    "steelblue",   "o"),
    ("shared",          "Shared SubGNN",      "darkorange",  "s"),
    ("attn_per_node",   "Per-node Attention", "mediumorchid","^"),
    ("attn_shared",     "Shared Attention",   "crimson",     "D"),
]


# ---------------------------------------------------------------------------
# OSPF baseline (heap-based queue simulation)
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
# Training + evaluation for one (D_r, variant) pair
# ---------------------------------------------------------------------------

def _build(variant_key: str, topo, cfg: ScaIRConfig) -> list:
    if variant_key == "per_node":
        return build_agents(topo, cfg)
    if variant_key == "shared":
        return build_agents_shared_gnn(topo, cfg)
    if variant_key == "attn_per_node":
        return build_agents(topo, cfg, gnn_cls=AttentionSubGNN)
    if variant_key == "attn_shared":
        return build_agents_shared_gnn(topo, cfg, gnn_cls=AttentionSubGNN)
    raise ValueError(f"Unknown variant: {variant_key}")


def run_variant(
    label: str,
    variant_key: str,
    topo, tms,
    cfg: ScaIRConfig,
) -> dict:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    is_shared = "shared" in variant_key
    agents = _build(variant_key, topo, cfg)
    env = RoutingEnvironment(topo, cfg)

    train_curve: List[float] = []
    t_start = time.time()

    for ep in range(1, TRAIN_EPISODES + 1):
        tm   = tms[(ep - 1) % len(tms)]
        pkts = env.generate_packets(tm, cfg.packets_per_episode)
        stats = env.run_episode(pkts, agents, training=True)
        if is_shared:
            agents[0].shared_gnn_step(len(agents))
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

    eval_times: List[float] = []
    for ep in range(1, EVAL_EPISODES + 1):
        tm   = tms[(TRAIN_EPISODES + ep - 1) % len(tms)]
        pkts = env.generate_packets(tm, cfg.packets_per_episode)
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
    drs = sorted(results.keys())
    ospf = [results[d]["ospf"]["avg_delivery_time"] for d in drs]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("SubGNN Variants vs OSPF: D_r Sweep (Abilene, 300 eps training)", fontsize=13)

    ax = axes[0]
    ax.plot(drs, ospf, "o--", color="green", label="OSPF", linewidth=2)
    for key, label, color, marker in VARIANTS:
        times = [results[d][key]["avg_delivery_time"] for d in drs]
        ax.plot(drs, times, f"{marker}-", color=color, label=label, linewidth=2)
    ax.set_xlabel("Hot-spot ratio D_r")
    ax.set_ylabel("Avg Delivery Time (ms)")
    ax.set_title("Delivery Time vs D_r")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xticks(drs)

    ax = axes[1]
    for key, label, color, marker in VARIANTS:
        gains = [(o - results[d][key]["avg_delivery_time"]) / max(o, 1e-6) * 100
                 for o, d in zip(ospf, drs)]
        ax.plot(drs, gains, f"{marker}-", color=color, label=label, linewidth=2)
    ax.axhline(0, color="green", linestyle="--", linewidth=1.5, label="OSPF parity")
    ax.set_xlabel("Hot-spot ratio D_r")
    ax.set_ylabel("Improvement over OSPF (%)")
    ax.set_title("Gain vs OSPF")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xticks(drs)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_training_curves(results: dict, out_path: str) -> None:
    drs = sorted(results.keys())
    n = len(drs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    fig.suptitle("Training Convergence by D_r", fontsize=13)
    if n == 1:
        axes = [axes]

    for i, (dr, ax) in enumerate(zip(drs, axes)):
        for key, label, color, marker in VARIANTS:
            curve = smooth(results[dr][key]["train_curve"])
            ax.plot(range(1, len(curve) + 1), curve, color=color, linewidth=1.5, label=label)
        ax.axhline(results[dr]["ospf"]["avg_delivery_time"], color="green",
                   linestyle="--", linewidth=1.2, label="OSPF")
        ax.set_title(f"D_r = {dr}")
        ax.set_xlabel("Episode")
        if i == 0:
            ax.set_ylabel("Avg Delivery Time (ms)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

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

    base_cfg = ScaIRConfig()
    max_deg  = max(len(v) for v in topo.adjacency.values())
    if topo.num_nodes > base_cfg.max_nodes:  base_cfg.max_nodes  = topo.num_nodes
    if max_deg        > base_cfg.max_degree: base_cfg.max_degree = max_deg

    all_results: dict = {}

    for dr in DR_VALUES:
        print(f"\n{'='*65}")
        print(f"  D_r = {dr}  ({TRAIN_EPISODES} training eps, {N_PACKETS} pkts/ep)")
        print(f"{'='*65}")

        import copy
        cfg = copy.copy(base_cfg)
        cfg.distribution_ratio  = dr
        cfg.packets_per_episode = N_PACKETS

        print(f"  OSPF baseline ({EVAL_EPISODES} eps)...")
        ospf = run_ospf(topo, cfg, tms, EVAL_EPISODES, N_PACKETS)
        print(f"    OSPF: {ospf['avg_delivery_time']:.3f} ms  "
              f"hops={ospf['avg_hops']:.2f}")

        dr_results: dict = {"ospf": ospf}
        for key, label, _, _ in VARIANTS:
            print(f"  Training {label}...")
            res = run_variant(f"D_r={dr} {label}", key, topo, tms, cfg)
            dr_results[key] = res
            o = ospf["avg_delivery_time"]
            gain = (o - res["avg_delivery_time"]) / max(o, 1e-6) * 100
            print(f"    {label}: {res['avg_delivery_time']:.3f} ms  ({gain:+.1f}% vs OSPF)")

        all_results[dr] = dr_results

    print("\nGenerating plots...")
    plot_dr_sweep(all_results, os.path.join(RESULTS, "comparison_dr_sweep.png"))
    plot_training_curves(all_results, os.path.join(RESULTS, "comparison_training_curves.png"))

    # Save JSON (drop train curves)
    json_out = {}
    for dr, res in all_results.items():
        json_out[str(dr)] = {
            "ospf": res["ospf"],
            **{k: {x: v for x, v in res[k].items() if x != "train_curve"}
               for k, _, _, _ in VARIANTS},
        }
    json_path = os.path.join(RESULTS, "comparison_experiment.json")
    with open(json_path, "w") as f:
        json.dump({"config": {"dr_values": DR_VALUES, "train_episodes": TRAIN_EPISODES,
                               "eval_episodes": EVAL_EPISODES, "n_packets": N_PACKETS,
                               "seed": SEED},
                   "results": json_out}, f, indent=2)
    print(f"  Saved: {json_path}")

    # Summary table
    header = f"{'D_r':>5}  {'OSPF':>8}" + "".join(f"  {k[0]:>12}  {'gain':>7}" for k, *_ in VARIANTS)
    print("\n" + "="*80)
    print(header)
    print("="*80)
    for dr in DR_VALUES:
        r = all_results[dr]
        o = r["ospf"]["avg_delivery_time"]
        row = f"  {dr:.1f}   {o:8.3f}"
        for key, *_ in VARIANTS:
            t = r[key]["avg_delivery_time"]
            g = (o - t) / max(o, 1e-6) * 100
            row += f"  {t:12.3f}  {g:+6.1f}%"
        print(row)
    print("="*80)


if __name__ == "__main__":
    main()
