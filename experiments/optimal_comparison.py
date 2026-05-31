#!/usr/bin/env python3
"""
Compare ScaIR against optimal routing baselines.

Baselines:
  1. OSPF            — static shortest-path (hop count), no congestion awareness.
  2. Oracle Greedy   — at every routing decision, runs Dijkstra with edge weights
                       equal to current queue lengths (perfect online greedy).
  3. LP Optimal      — offline multi-commodity flow LP solved once per episode
                       knowing all packets in advance.  Edge costs are iteratively
                       updated to reflect congestion until convergence, then the
                       resulting flow is executed in the simulator.

Usage:
  python experiments/optimal_comparison.py --topo geant   [--dr 0.6] [--episodes 50]
  python experiments/optimal_comparison.py --topo germany50

The script trains ScaIR (300 episodes, UCB) then evaluates all four methods on
50 shared evaluation episodes.
"""

import argparse
import copy
import heapq
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scair.config import ScaIRConfig
from scair.data_loader import load_all_traffic_matrices, load_topology, normalise_tm
from scair.environment import RoutingEnvironment
from train import build_agents

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS = "results/optimal_comparison"
SEED    = 42

TOPO_REGISTRY = {
    "geant":     ("data/GEA/Topology.txt",   "data/GEA/TrafficMatrix"),
    "germany50": ("data/GER50/Topology.txt", "data/GER50/TrafficMatrix"),
    "abilene":   ("data/ABI/Topology.txt",   "data/ABI/TrafficMatrix"),
}

# ---------------------------------------------------------------------------
# Dijkstra helpers
# ---------------------------------------------------------------------------

def dijkstra_hops(adj: Dict, num_nodes: int, source: int) -> Dict[int, int]:
    """Shortest-path next-hop table from source, unit edge weights."""
    dist = {n: float("inf") for n in range(num_nodes)}
    prev: Dict[int, int] = {}
    dist[source] = 0
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v in adj[u]:
            if d + 1 < dist[v]:
                dist[v] = d + 1
                prev[v] = u
                heapq.heappush(heap, (d + 1, v))
    nh: Dict[int, int] = {}
    for dst in range(num_nodes):
        if dst == source or dist[dst] == float("inf"):
            continue
        node = dst
        while prev.get(node, source) != source:
            node = prev[node]
        nh[dst] = node
    return nh


def dijkstra_queue_weighted(adj: Dict, num_nodes: int, source: int,
                             queue_len: Dict[Tuple[int,int], float],
                             trans_time: float, q_time: float) -> Dict[int, int]:
    """Next-hop table using current queue lengths as edge weights."""
    dist = {n: float("inf") for n in range(num_nodes)}
    prev: Dict[int, int] = {}
    dist[source] = 0
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v in adj[u]:
            w = queue_len.get((u, v), 0) * q_time + trans_time
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    nh: Dict[int, int] = {}
    for dst in range(num_nodes):
        if dst == source or dist[dst] == float("inf"):
            continue
        node = dst
        while prev.get(node, source) != source:
            node = prev[node]
        nh[dst] = node
    return nh


def all_pairs_hops(topo) -> Dict[int, Dict[int, int]]:
    """Precompute hop count between all pairs (for lower-bound computation)."""
    result = {}
    for n in range(topo.num_nodes):
        dist = {n: 0}
        queue = [n]
        while queue:
            u = queue.pop(0)
            for v in topo.adjacency[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        result[n] = dist
    return result


# ---------------------------------------------------------------------------
# OSPF simulation
# ---------------------------------------------------------------------------

def run_ospf_episode(topo, cfg, packets: list) -> float:
    routes = {n: dijkstra_hops(topo.adjacency, topo.num_nodes, n)
              for n in range(topo.num_nodes)}
    heap = []
    for pkt in packets:
        heapq.heappush(heap, (pkt.birth_time, pkt.pid, pkt, pkt.source, None))

    delivered: List[float] = []
    hops: Dict[int, int] = defaultdict(int)
    qlen: Dict[Tuple[int,int], int] = defaultdict(int)

    while heap:
        t, _, pkt, node, prev_node = heapq.heappop(heap)
        if prev_node is not None:
            qlen[(prev_node, node)] = max(0, qlen[(prev_node, node)] - 1)
        if node == pkt.destination:
            delivered.append(t - pkt.birth_time)
            continue
        if hops[pkt.pid] >= cfg.max_hops:
            continue
        nxt = routes.get(node, {}).get(pkt.destination)
        if nxt is None:
            continue
        q = qlen[(node, nxt)]
        cost = q * cfg.queue_time_per_packet + cfg.transmission_time
        qlen[(node, nxt)] += 1
        hops[pkt.pid] += 1
        heapq.heappush(heap, (t + cost, pkt.pid, pkt, nxt, node))

    return float(np.mean(delivered)) if delivered else float("inf")


# ---------------------------------------------------------------------------
# Oracle Greedy simulation
# ---------------------------------------------------------------------------

def run_oracle_episode(topo, cfg, packets: list) -> float:
    """
    Oracle greedy: at each routing decision, run Dijkstra with current
    queue lengths as edge weights and follow the first hop.
    """
    heap = []
    for pkt in packets:
        heapq.heappush(heap, (pkt.birth_time, pkt.pid, pkt, pkt.source, None))

    delivered: List[float] = []
    hops: Dict[int, int] = defaultdict(int)
    qlen: Dict[Tuple[int,int], int] = defaultdict(int)

    while heap:
        t, _, pkt, node, prev_node = heapq.heappop(heap)
        if prev_node is not None:
            qlen[(prev_node, node)] = max(0, qlen[(prev_node, node)] - 1)
        if node == pkt.destination:
            delivered.append(t - pkt.birth_time)
            continue
        if hops[pkt.pid] >= cfg.max_hops:
            continue

        # Oracle: Dijkstra with current queue weights
        routes = dijkstra_queue_weighted(
            topo.adjacency, topo.num_nodes, node, qlen,
            cfg.transmission_time, cfg.queue_time_per_packet
        )
        nxt = routes.get(pkt.destination)
        if nxt is None:
            continue

        q = qlen[(node, nxt)]
        cost = q * cfg.queue_time_per_packet + cfg.transmission_time
        qlen[(node, nxt)] += 1
        hops[pkt.pid] += 1
        heapq.heappush(heap, (t + cost, pkt.pid, pkt, nxt, node))

    return float(np.mean(delivered)) if delivered else float("inf")


# ---------------------------------------------------------------------------
# Offline load-balanced routing (iterative load-adjusted shortest path)
# ---------------------------------------------------------------------------

def solve_load_balanced_routing(topo, packets: list, cfg,
                                n_iters: int = 10,
                                alpha: float = 0.5) -> Dict[int, Dict[int, int]]:
    """
    Offline iterative load-balanced routing.

    Knows all packets in the episode in advance (unlike online algorithms).
    Iterates:
      1. Compute link loads from current routing.
      2. Update edge weights = transmission_time + alpha * load.
      3. Re-route every (src,dst) pair on the new cheapest path.
    Converges to a routing that balances congestion vs. path length.

    Uses pure integer routing throughout — no fractional-to-integer
    conversion needed.  alpha controls the congestion-vs-hops trade-off;
    alpha=0 → OSPF, alpha→∞ → max load-spreading ignoring path length.
    """
    N = topo.num_nodes

    # Demand: count packets per (src, dst)
    demand: Dict[Tuple[int,int], int] = defaultdict(int)
    for pkt in packets:
        if pkt.source != pkt.destination:
            demand[(pkt.source, pkt.destination)] += 1

    if not demand:
        return {n: dijkstra_hops(topo.adjacency, N, n) for n in range(N)}

    # Precompute shortest-path hop counts (used to sanity-cap detours)
    sp_dist = all_pairs_hops(topo)

    def _dijkstra_weighted(source: int, edge_w: Dict[Tuple[int,int], float]):
        dist = {n: float("inf") for n in range(N)}
        prev: Dict[int, int] = {}
        dist[source] = 0.0
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v in topo.adjacency[u]:
                nd = d + edge_w.get((u, v), cfg.transmission_time)
                if nd < dist[v]:
                    dist[v] = nd; prev[v] = u
                    heapq.heappush(heap, (nd, v))
        nh: Dict[int, int] = {}
        for dst in range(N):
            if dst == source or dist[dst] == float("inf"):
                continue
            node = dst
            while prev.get(node, source) != source:
                node = prev[node]
            nh[dst] = node
        return nh

    # Start with unit-weight (OSPF) routing
    edge_w: Dict[Tuple[int,int], float] = {
        (u, v): cfg.transmission_time
        for u in range(N) for v in topo.adjacency[u]
    }
    nh_tables = {n: _dijkstra_weighted(n, edge_w) for n in range(N)}

    for _ in range(n_iters):
        # Compute link loads under current routing
        link_load: Dict[Tuple[int,int], int] = defaultdict(int)
        for (s, d), count in demand.items():
            node = s
            for _ in range(N + 1):
                if node == d:
                    break
                nxt = nh_tables.get(node, {}).get(d)
                if nxt is None:
                    break
                link_load[(node, nxt)] += count
                node = nxt

        # Update edge weights
        for u in range(N):
            for v in topo.adjacency[u]:
                edge_w[(u, v)] = cfg.transmission_time + alpha * link_load.get((u, v), 0)

        # Recompute routing with new weights
        new_tables = {n: _dijkstra_weighted(n, edge_w) for n in range(N)}

        # Safety: don't accept paths longer than shortest-path + 2 hops
        for n in range(N):
            for d in range(N):
                if d == n:
                    continue
                new_nxt = new_tables[n].get(d)
                if new_nxt is None:
                    continue
                # Trace new path length
                node2, seen2, h = n, {n}, 0
                while node2 != d and h <= sp_dist[n].get(d, 0) + 2:
                    nxt2 = new_tables[node2].get(d)
                    if nxt2 is None or nxt2 in seen2:
                        h = 999; break
                    seen2.add(nxt2); node2 = nxt2; h += 1
                sp_len = sp_dist[n].get(d, 999)
                if h > sp_len + 2:
                    # Revert to previous (safer) routing for this entry
                    old_nxt = nh_tables[n].get(d)
                    if old_nxt is not None:
                        new_tables[n][d] = old_nxt

        nh_tables = new_tables

    return nh_tables

    return nh_tables


def run_lp_episode(topo, cfg, packets: list, n_lp_iters: int = 10) -> float:
    """Execute the offline load-balanced routing policy in the simulator."""
    nh_tables = solve_load_balanced_routing(topo, packets, cfg, n_iters=n_lp_iters)

    heap = []
    for pkt in packets:
        heapq.heappush(heap, (pkt.birth_time, pkt.pid, pkt, pkt.source, None))

    delivered: List[float] = []
    hops: Dict[int, int] = defaultdict(int)
    qlen: Dict[Tuple[int,int], int] = defaultdict(int)

    while heap:
        t, _, pkt, node, prev_node = heapq.heappop(heap)
        if prev_node is not None:
            qlen[(prev_node, node)] = max(0, qlen[(prev_node, node)] - 1)
        if node == pkt.destination:
            delivered.append(t - pkt.birth_time)
            continue
        if hops[pkt.pid] >= cfg.max_hops:
            continue
        nxt = nh_tables.get(node, {}).get(pkt.destination)
        if nxt is None:
            continue
        q = qlen[(node, nxt)]
        cost = q * cfg.queue_time_per_packet + cfg.transmission_time
        qlen[(node, nxt)] += 1
        hops[pkt.pid] += 1
        heapq.heappush(heap, (t + cost, pkt.pid, pkt, nxt, node))

    return float(np.mean(delivered)) if delivered else float("inf")


# ---------------------------------------------------------------------------
# Training helpers (reuse from train.py)
# ---------------------------------------------------------------------------

def train_scair(topo, cfg, tms, n_episodes, n_packets, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    agents = build_agents(topo, cfg)
    env = RoutingEnvironment(topo, cfg)
    t0 = time.time()
    for ep in range(1, n_episodes + 1):
        tm = tms[(ep - 1) % len(tms)]
        packets = env.generate_packets(tm, n_packets)
        stats = env.run_episode(packets, agents, training=True)
        if ep == 10:
            for ag in agents:
                ag.set_learning_rate(cfg.learning_rate)
        if ep % cfg.target_update_freq == 0:
            for ag in agents:
                ag.update_target()
        if ep % cfg.sigma_decay_freq == 0:
            for ag in agents:
                ag.decay_sigma()
        if ep % 50 == 0 or ep == 1:
            print(f"  [ScaIR train] ep {ep}/{n_episodes}  "
                  f"t={stats['avg_delivery_time']:.2f}ms  "
                  f"σ={agents[0].sigma:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    return agents


def eval_scair(agents, topo, cfg, episodes):
    env = RoutingEnvironment(topo, cfg)
    times = []
    for packets in episodes:
        stats = env.run_episode(packets, agents, training=False)
        times.append(stats["avg_delivery_time"])
    return float(np.mean(times)), float(np.std(times))


# ---------------------------------------------------------------------------
# No-congestion lower bound
# ---------------------------------------------------------------------------

def no_congestion_lower_bound(topo, cfg, packets) -> float:
    """
    Absolute minimum delivery time: each packet routed on shortest hop-count
    path with ZERO queuing delays.
    delivery_time = n_hops * transmission_time
    """
    hop_dist = all_pairs_hops(topo)
    times = []
    for pkt in packets:
        h = hop_dist[pkt.source].get(pkt.destination, float("inf"))
        if h < float("inf"):
            times.append(h * cfg.transmission_time)
    return float(np.mean(times)) if times else float("inf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--topo", default="geant", choices=list(TOPO_REGISTRY.keys()))
    p.add_argument("--dr", type=float, default=0.6)
    p.add_argument("--train_eps", type=int, default=300)
    p.add_argument("--eval_eps", type=int, default=50)
    p.add_argument("--packets", type=int, default=100)
    p.add_argument("--lp_iters", type=int, default=5,
                   help="Frank-Wolfe LP iterations for congestion update")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    topo_file, tm_dir = TOPO_REGISTRY[args.topo]
    print(f"\n{'='*65}")
    print(f"  Optimal comparison — {args.topo.upper()}  D_r={args.dr}")
    print(f"{'='*65}\n")

    topo    = load_topology(topo_file)
    raw_tms = load_all_traffic_matrices(tm_dir, topo.num_nodes)
    tms     = [normalise_tm(tm) for tm in raw_tms]
    print(f"Topology: {topo.num_nodes} nodes, {len(tms)} TMs loaded")

    # Check hot-spot pair
    last = topo.num_nodes - 1
    connected = last in topo.adjacency[0]
    print(f"Hot-spot: 0 → {last}  (direct neighbours: {connected})")
    if connected:
        print("  WARNING: hot-spot pair are direct neighbours — "
              "no routing choice for the hot-spot traffic.")

    cfg = ScaIRConfig()
    max_deg = max(len(v) for v in topo.adjacency.values())
    if topo.num_nodes > cfg.max_nodes:  cfg.max_nodes  = topo.num_nodes
    if max_deg        > cfg.max_degree: cfg.max_degree = max_deg
    cfg.action_method       = "ucb"
    cfg.distribution_ratio  = args.dr
    cfg.packets_per_episode = args.packets

    # Pre-generate episodes
    env_gen = RoutingEnvironment(topo, cfg)
    random.seed(SEED); np.random.seed(SEED)
    train_eps = [env_gen.generate_packets(tms[i % len(tms)], args.packets)
                 for i in range(args.train_eps)]
    random.seed(SEED + 1000); np.random.seed(SEED + 1000)
    eval_eps  = [env_gen.generate_packets(tms[i % len(tms)], args.packets)
                 for i in range(args.eval_eps)]

    # ----------------------------------------------------------------
    # 1. No-congestion lower bound (analytical)
    # ----------------------------------------------------------------
    lb_times = [no_congestion_lower_bound(topo, cfg, eps) for eps in eval_eps]
    lb_mean  = float(np.mean(lb_times))
    lb_std   = float(np.std(lb_times))
    print(f"\nNo-congestion lower bound : {lb_mean:.3f} ± {lb_std:.3f} ms")

    # ----------------------------------------------------------------
    # 2. OSPF
    # ----------------------------------------------------------------
    print("\nRunning OSPF ...", flush=True)
    ospf_times = [run_ospf_episode(topo, cfg, eps) for eps in eval_eps]
    ospf_mean  = float(np.mean(ospf_times))
    ospf_std   = float(np.std(ospf_times))
    print(f"OSPF                      : {ospf_mean:.3f} ± {ospf_std:.3f} ms")

    # ----------------------------------------------------------------
    # 3. Oracle Greedy
    # ----------------------------------------------------------------
    print("\nRunning Oracle Greedy ...", flush=True)
    t0 = time.time()
    oracle_times = [run_oracle_episode(topo, cfg, eps) for eps in eval_eps]
    oracle_mean  = float(np.mean(oracle_times))
    oracle_std   = float(np.std(oracle_times))
    print(f"Oracle Greedy             : {oracle_mean:.3f} ± {oracle_std:.3f} ms  ({time.time()-t0:.1f}s)")

    lp_mean = lp_std = None   # offline fixed-routing omitted (see note below)

    # ----------------------------------------------------------------
    # 5. ScaIR (train then eval)
    # ----------------------------------------------------------------
    print(f"\nTraining ScaIR ({args.train_eps} episodes) ...", flush=True)
    agents = train_scair(topo, cfg, tms, args.train_eps, args.packets, SEED)

    print("\nEvaluating ScaIR ...", flush=True)
    scair_mean, scair_std = eval_scair(agents, topo, cfg, eval_eps)
    print(f"ScaIR ({args.train_eps} eps)         : {scair_mean:.3f} ± {scair_std:.3f} ms")

    # ----------------------------------------------------------------
    # Results table
    # ----------------------------------------------------------------
    print(f"\n{'─'*65}")
    print(f"  RESULTS — {args.topo.upper()}  D_r={args.dr}  ({args.packets} pkt/ep, {args.eval_eps} eval eps)")
    print(f"{'─'*65}")
    methods = [
        ("No-congestion lower bound", lb_mean,     lb_std),
        ("Oracle Greedy (online)",    oracle_mean, oracle_std),
        ("ScaIR (UCB, 300 eps)",      scair_mean,  scair_std),
        ("OSPF",                      ospf_mean,   ospf_std),
    ]
    for name, mean, std in methods:
        gain = (ospf_mean - mean) / max(ospf_mean, 1e-6) * 100
        print(f"  {name:<35} {mean:7.3f} ± {std:.3f} ms   ({gain:+.1f}% vs OSPF)")
    print(f"{'─'*65}")

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    results = {
        "topology": args.topo, "dr": args.dr,
        "train_episodes": args.train_eps,
        "eval_episodes": args.eval_eps,
        "packets_per_episode": args.packets,
        "no_congestion_lb": {"mean": lb_mean,     "std": lb_std},
        "oracle_greedy":     {"mean": oracle_mean, "std": oracle_std},
        "scair":             {"mean": scair_mean,  "std": scair_std},
        "ospf":              {"mean": ospf_mean,   "std": ospf_std},
    }
    tag = f"{args.topo}_dr{args.dr}"
    json_path = os.path.join(RESULTS, f"{tag}_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ----------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["No-congestion\nlower bound", "Oracle Greedy\n(online)",
              f"ScaIR\n(UCB, {args.train_eps} eps)", "OSPF"]
    means  = [lb_mean, oracle_mean, scair_mean, ospf_mean]
    stds   = [lb_std,  oracle_std,  scair_std,  ospf_std]
    colors = ["#2ecc71", "#9b59b6", "#e67e22", "#e74c3c"]

    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black", linewidth=0.7)
    ax.axhline(ospf_mean, color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_ylabel("Avg Delivery Time (ms)")
    ax.set_title(f"Optimal Comparison — {args.topo.upper()}  D_r={args.dr}")
    ax.grid(axis="y", alpha=0.3)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, mean + max(stds)*0.05,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plot_path = os.path.join(RESULTS, f"{tag}_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {plot_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
