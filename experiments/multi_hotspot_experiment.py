#!/usr/bin/env python3
"""
Multi-hotspot routing experiment on Germany50.

Four simultaneous hot-spot pairs, chosen so their shortest paths share
intermediate high-betweenness nodes, creating genuine load-balancing
competition that cannot be solved by simply avoiding one corridor.

Hot-spot pairs (each gets DR_PER_PAIR = DR_TOTAL/4 of packets):
  5  → 24  (5 hops, path: 5-4-44-28-23-24)
  49 → 28  (3 hops, path: 49-18-16-28)         ← shares node 28 with pair 1
  25 → 45  (3 hops, path: 25-13-49-45)         ← shares node 49 with pair 2
  13 → 22  (3 hops, path: 13-25-5-22)          ← shares nodes 25,5 with pairs 1,3

Compares:
  1. No-congestion lower bound (zero-queue floor)
  2. Oracle Greedy  — Dijkstra with live queue weights at every decision
  3. ScaIR (UCB, 300 episodes)
  4. OSPF (static shortest-path)
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
from scair.environment import RoutingEnvironment, Packet
from experiments.optimal_comparison import (
    dijkstra_hops, dijkstra_queue_weighted, all_pairs_hops,
    run_oracle_episode, run_ospf_episode,
    train_scair, eval_scair,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS  = "results/multi_hotspot"
SEED     = 42
TOPO_FILE = "data/GER50/Topology.txt"
TM_DIR    = "data/GER50/TrafficMatrix"

# 4 hot-spot pairs (src, dst) — paths share nodes 49, 28, 25, 13, 5
HOTSPOT_PAIRS = [(5, 24), (49, 28), (25, 45), (13, 22)]
DR_TOTAL   = 0.6          # total fraction of hot-spot packets
DR_PER_PAIR = DR_TOTAL / len(HOTSPOT_PAIRS)   # 0.15 each

TRAIN_EPISODES = 300
EVAL_EPISODES  = 50
N_PACKETS      = 100


# ---------------------------------------------------------------------------
# Multi-hotspot packet generator
# ---------------------------------------------------------------------------

def generate_multi_hotspot(topo, tm: np.ndarray, n_packets: int,
                            hotspot_pairs: List[Tuple[int,int]],
                            dr_total: float,
                            gen_interval: float) -> List[Packet]:
    """
    Generate packets with multiple hot-spot pairs.

    Each hot-spot pair gets an equal share of dr_total traffic.
    Remaining (1 - dr_total) fraction is sampled proportional to TM.
    """
    n = topo.num_nodes
    K = len(hotspot_pairs)
    dr_per = dr_total / K

    # TM probabilities (diagonal zeroed)
    tm_f = tm.astype(float).copy()
    np.fill_diagonal(tm_f, 0.0)
    total = tm_f.sum()
    if total > 0:
        probs = (tm_f / total).flatten()
    else:
        probs = np.ones(n * n) / max(n * n - n, 1)
        for i in range(n):
            probs[i * n + i] = 0.0

    packets: List[Packet] = []
    t = 0.0
    for pid in range(n_packets):
        t += np.random.exponential(gen_interval)
        r = random.random()
        if r < dr_total:
            # Which hot-spot pair?
            pair_idx = int(r / dr_per) % K
            src, dst = hotspot_pairs[pair_idx]
        else:
            idx = int(np.random.choice(n * n, p=probs))
            src, dst = divmod(idx, n)
        packets.append(Packet(pid, src, dst, t))

    return packets


def pregenerate_multi_hotspot(topo, tms, n_episodes, n_packets,
                               hotspot_pairs, dr_total, cfg, seed):
    random.seed(seed); np.random.seed(seed)
    return [
        generate_multi_hotspot(
            topo, tms[i % len(tms)], n_packets,
            hotspot_pairs, dr_total, cfg.generation_interval
        )
        for i in range(n_episodes)
    ]


# ---------------------------------------------------------------------------
# No-congestion lower bound (hot-spot aware)
# ---------------------------------------------------------------------------

def no_congestion_lb(topo, cfg, packets: List[Packet]) -> float:
    hop_dist = all_pairs_hops(topo)
    times = []
    for pkt in packets:
        h = hop_dist[pkt.source].get(pkt.destination, None)
        if h is not None:
            times.append(h * cfg.transmission_time)
    return float(np.mean(times)) if times else float("inf")


# ---------------------------------------------------------------------------
# ScaIR training with multi-hotspot traffic
# ---------------------------------------------------------------------------

def train_scair_multi(topo, cfg, tms, n_episodes, n_packets,
                      hotspot_pairs, dr_total, seed):
    """Train ScaIR agents using multi-hotspot episodes."""
    from train import build_agents

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    agents = build_agents(topo, cfg)
    env = RoutingEnvironment(topo, cfg)
    t0 = time.time()

    for ep in range(1, n_episodes + 1):
        random.seed(seed + ep); np.random.seed(seed + ep)
        tm = tms[(ep - 1) % len(tms)]
        packets = generate_multi_hotspot(
            topo, tm, n_packets, hotspot_pairs, dr_total, cfg.generation_interval
        )
        stats = env.run_episode(packets, agents, training=True)

        if ep == 10:
            for ag in agents: ag.set_learning_rate(cfg.learning_rate)
        if ep % cfg.target_update_freq == 0:
            for ag in agents: ag.update_target()
        if ep % cfg.sigma_decay_freq == 0:
            for ag in agents: ag.decay_sigma()
        if ep % 50 == 0 or ep == 1:
            print(f"  [ScaIR train] ep {ep}/{n_episodes}  "
                  f"t={stats['avg_delivery_time']:.2f}ms  "
                  f"σ={agents[0].sigma:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    return agents


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Multi-Hotspot Experiment — Germany50")
    print(f"  {len(HOTSPOT_PAIRS)} hot-spot pairs, D_r={DR_TOTAL} total ({DR_PER_PAIR:.2f}/pair)")
    for s, d in HOTSPOT_PAIRS:
        print(f"    {s} → {d}")
    print(f"{'='*65}\n")

    topo    = load_topology(TOPO_FILE)
    raw_tms = load_all_traffic_matrices(TM_DIR, topo.num_nodes)
    tms     = [normalise_tm(tm) for tm in raw_tms]
    print(f"Topology: {topo.num_nodes} nodes, {len(tms)} TMs")

    cfg = ScaIRConfig()
    max_deg = max(len(v) for v in topo.adjacency.values())
    if topo.num_nodes > cfg.max_nodes:  cfg.max_nodes  = topo.num_nodes
    if max_deg        > cfg.max_degree: cfg.max_degree = max_deg
    cfg.action_method       = "ucb"
    cfg.distribution_ratio  = 0.0      # handled manually
    cfg.packets_per_episode = N_PACKETS

    # Pre-generate episodes (fixed seed → all methods see identical traffic)
    print("Pre-generating episodes...", flush=True)
    train_eps = pregenerate_multi_hotspot(
        topo, tms, TRAIN_EPISODES, N_PACKETS,
        HOTSPOT_PAIRS, DR_TOTAL, cfg, seed=SEED
    )
    eval_eps = pregenerate_multi_hotspot(
        topo, tms, EVAL_EPISODES, N_PACKETS,
        HOTSPOT_PAIRS, DR_TOTAL, cfg, seed=SEED + 1000
    )

    # Verify hot-spot fraction in eval episodes
    hs_set = set(HOTSPOT_PAIRS)
    hs_count = sum(1 for ep in eval_eps for p in ep if (p.source, p.destination) in hs_set)
    total_pkts = sum(len(ep) for ep in eval_eps)
    print(f"  Hot-spot fraction in eval: {hs_count/total_pkts:.2%} (target {DR_TOTAL:.0%})")

    # ----------------------------------------------------------------
    # 1. No-congestion lower bound
    # ----------------------------------------------------------------
    lb_times = [no_congestion_lb(topo, cfg, ep) for ep in eval_eps]
    lb_mean  = float(np.mean(lb_times))
    lb_std   = float(np.std(lb_times))
    print(f"\nNo-congestion lower bound : {lb_mean:.3f} ± {lb_std:.3f} ms")

    # ----------------------------------------------------------------
    # 2. OSPF
    # ----------------------------------------------------------------
    print("Running OSPF ...", flush=True)
    ospf_times = [run_ospf_episode(topo, cfg, ep) for ep in eval_eps]
    ospf_mean  = float(np.mean(ospf_times))
    ospf_std   = float(np.std(ospf_times))
    print(f"OSPF                      : {ospf_mean:.3f} ± {ospf_std:.3f} ms")

    # ----------------------------------------------------------------
    # 3. Oracle Greedy
    # ----------------------------------------------------------------
    print("Running Oracle Greedy ...", flush=True)
    t0 = time.time()
    oracle_times = [run_oracle_episode(topo, cfg, ep) for ep in eval_eps]
    oracle_mean  = float(np.mean(oracle_times))
    oracle_std   = float(np.std(oracle_times))
    print(f"Oracle Greedy             : {oracle_mean:.3f} ± {oracle_std:.3f} ms  "
          f"({time.time()-t0:.1f}s)")

    # ----------------------------------------------------------------
    # 4. ScaIR
    # ----------------------------------------------------------------
    print(f"\nTraining ScaIR ({TRAIN_EPISODES} eps) ...", flush=True)
    agents = train_scair_multi(
        topo, cfg, tms, TRAIN_EPISODES, N_PACKETS,
        HOTSPOT_PAIRS, DR_TOTAL, SEED
    )
    print("Evaluating ScaIR ...", flush=True)
    env_eval = RoutingEnvironment(topo, cfg)
    scair_times = []
    for ep in eval_eps:
        stats = env_eval.run_episode(ep, agents, training=False)
        scair_times.append(stats["avg_delivery_time"])
    scair_mean = float(np.mean(scair_times))
    scair_std  = float(np.std(scair_times))
    print(f"ScaIR (UCB, {TRAIN_EPISODES} eps)     : {scair_mean:.3f} ± {scair_std:.3f} ms")

    # ----------------------------------------------------------------
    # Results
    # ----------------------------------------------------------------
    print(f"\n{'─'*65}")
    print(f"  RESULTS — Germany50  Multi-hotspot D_r={DR_TOTAL}")
    print(f"  {len(HOTSPOT_PAIRS)} pairs: {HOTSPOT_PAIRS}")
    print(f"{'─'*65}")
    methods = [
        ("No-congestion lower bound", lb_mean,     lb_std),
        ("Oracle Greedy (online)",    oracle_mean, oracle_std),
        ("ScaIR (UCB, 300 eps)",      scair_mean,  scair_std),
        ("OSPF",                      ospf_mean,   ospf_std),
    ]
    for name, mean, std in methods:
        gain = (ospf_mean - mean) / max(ospf_mean, 1e-6) * 100
        print(f"  {name:<35} {mean:7.3f} ± {std:.3f} ms  ({gain:+.1f}% vs OSPF)")
    print(f"{'─'*65}")

    oracle_gap = (scair_mean - oracle_mean) / max(oracle_mean, 1e-6) * 100
    irred      = (oracle_mean - lb_mean)    / max(ospf_mean - lb_mean, 1e-6) * 100
    scair_sub  = (scair_mean  - oracle_mean)/ max(ospf_mean - lb_mean, 1e-6) * 100
    print(f"\n  ScaIR vs Oracle gap       : +{oracle_gap:.1f}%")
    print(f"  Irreducible congestion    : {irred:.1f}% of OSPF→floor gap")
    print(f"  ScaIR suboptimality       : {scair_sub:.1f}% of OSPF→floor gap")

    # Compare to single-hotspot (from stored results if available)
    sh_path = "results/optimal_comparison/germany50_dr0.6_results.json"
    if os.path.exists(sh_path):
        sh = json.load(open(sh_path))
        print(f"\n  ── vs single-hotspot (D_r=0.6) ──")
        print(f"  ScaIR gap: single={sh['scair']['mean']:.3f}ms → multi={scair_mean:.3f}ms "
              f"({'harder' if scair_mean > sh['scair']['mean'] else 'easier'})")
        print(f"  Oracle gap: single={sh['oracle_greedy']['mean']:.3f}ms → "
              f"multi={oracle_mean:.3f}ms "
              f"({'harder' if oracle_mean > sh['oracle_greedy']['mean'] else 'easier'})")
        sh_gap = (sh['scair']['mean'] - sh['oracle_greedy']['mean']) / \
                  max(sh['oracle_greedy']['mean'], 1e-6) * 100
        print(f"  ScaIR vs Oracle: single={sh_gap:.1f}% → multi={oracle_gap:.1f}%")

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    results = {
        "topology": "germany50",
        "hotspot_pairs": HOTSPOT_PAIRS,
        "dr_total": DR_TOTAL,
        "dr_per_pair": DR_PER_PAIR,
        "train_episodes": TRAIN_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "packets_per_episode": N_PACKETS,
        "no_congestion_lb": {"mean": lb_mean,     "std": lb_std},
        "oracle_greedy":     {"mean": oracle_mean, "std": oracle_std},
        "scair":             {"mean": scair_mean,  "std": scair_std},
        "ospf":              {"mean": ospf_mean,   "std": ospf_std},
    }
    json_path = os.path.join(RESULTS, "germany50_multi_hotspot_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ----------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Germany50 — Multi-Hotspot (4 pairs, D_r=0.6 total)", fontsize=13)

    # Bar chart
    ax = axes[0]
    labels = ["No-congestion\nlower bound", "Oracle Greedy\n(online)",
              f"ScaIR\n(UCB, {TRAIN_EPISODES} eps)", "OSPF"]
    means  = [lb_mean, oracle_mean, scair_mean, ospf_mean]
    stds   = [lb_std,  oracle_std,  scair_std,  ospf_std]
    colors = ["#2ecc71", "#9b59b6", "#e67e22", "#e74c3c"]
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors,
                  alpha=0.85, edgecolor="black", linewidth=0.7)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, mean + max(stds)*0.1,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Avg Delivery Time (ms)")
    ax.set_title("Average Delivery Time")
    ax.grid(axis="y", alpha=0.3)

    # Gap decomposition — horizontal stacked bar (handles negative ScaIR-vs-OSPF)
    ax = axes[1]
    ospf_floor = ospf_mean - lb_mean
    segments = [
        ("Irreducible congestion\n(Oracle − LB)",  oracle_mean - lb_mean,   "#95a5a6"),
        ("ScaIR suboptimality\n(ScaIR − Oracle)",  scair_mean  - oracle_mean, "#e67e22"),
        ("ScaIR vs OSPF\n(OSPF − ScaIR)",          ospf_mean   - scair_mean, "#2ecc71"),
    ]
    y, left = 0, lb_mean
    for label, width, color in segments:
        ax.barh(y, width, left=left, color=color, edgecolor="white", linewidth=1.5, height=0.5)
        if abs(width) > 0.05:
            ax.text(left + width/2, y, f"{width:+.2f}ms", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
        left += width
    ax.set_yticks([])
    ax.set_xlabel("Delivery Time (ms)")
    ax.axvline(lb_mean,     color="#2ecc71", linestyle="--", linewidth=1.2, label=f"LB {lb_mean:.2f}ms")
    ax.axvline(oracle_mean, color="#9b59b6", linestyle="--", linewidth=1.2, label=f"Oracle {oracle_mean:.2f}ms")
    ax.axvline(scair_mean,  color="#e67e22", linestyle="--", linewidth=1.2, label=f"ScaIR {scair_mean:.2f}ms")
    ax.axvline(ospf_mean,   color="#e74c3c", linestyle="--", linewidth=1.2, label=f"OSPF {ospf_mean:.2f}ms")
    ax.legend(fontsize=8, loc="upper right")
    pct = lambda v: f"{v/max(ospf_floor,1e-6)*100:.0f}%"
    title_parts = [f"{s[0].split(chr(10))[0]}: {pct(s[1])}" for s in segments]
    ax.set_title("Gap decomposition\n" + " | ".join(title_parts), fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS, "germany50_multi_hotspot_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {plot_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
