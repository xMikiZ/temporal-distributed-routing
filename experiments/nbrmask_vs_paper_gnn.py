#!/usr/bin/env python3
"""
Controlled comparison: NbrMask (no-GNN) vs Paper GNN on Abilene.

All episode packets are pre-generated once from a fixed seed so both
variants face byte-for-byte identical traffic. Only agent weights differ.

Variants
--------
  nbrmask  -- NeighborMaskSubGNN, per-node Q, UCB (no learning in GNN)
  paper    -- PaperSubGNN (mean_y f_w(V_y)), per-node Q, UCB

Usage
-----
  python experiments/nbrmask_vs_paper_gnn.py

Outputs
-------
  results/08_nbrmask_vs_paper/abilene_dr_sweep.png
  results/08_nbrmask_vs_paper/abilene_training_curves.png
  results/08_nbrmask_vs_paper/results.json
"""

import copy
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
from scair.models import PaperSubGNN, NeighborMaskSubGNN
from train import build_agents, build_agents_no_gnn

RESULTS        = "results/08_nbrmask_vs_paper"
SEED           = 42
DR_VALUES      = [0.0, 0.2, 0.4, 0.6, 0.8]
TRAIN_EPISODES = 300
EVAL_EPISODES  = 50
N_PACKETS      = 100
LOG_INTERVAL   = 100

TOPO_FILE = "data/ABI/Topology.txt"
TM_DIR    = "data/ABI/TrafficMatrix"

VARIANTS = [
    ("nbrmask", "NbrMask (no GNN)",      "darkorange", "^"),
    ("paper",   "Paper GNN (mean f_w)", "crimson",    "s"),
]


# ---------------------------------------------------------------------------
# Pre-generate episodes (fixes environment randomness across variants)
# ---------------------------------------------------------------------------

def pregenerate_episodes(env: RoutingEnvironment, tms, n_train: int, n_eval: int,
                          n_packets: int, seed: int) -> List:
    """Generate train+eval packet batches from a fixed seed. Returns list of packet lists."""
    random.seed(seed)
    np.random.seed(seed)
    episodes = []
    total = n_train + n_eval
    for ep in range(total):
        tm = tms[ep % len(tms)]
        episodes.append(env.generate_packets(tm, n_packets))
    return episodes


# ---------------------------------------------------------------------------
# OSPF baseline
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


def run_ospf(topo, cfg: ScaIRConfig, eval_episodes: List) -> dict:
    routes = {n: _dijkstra_next_hop(topo, n) for n in range(topo.num_nodes)}
    env = RoutingEnvironment(topo, cfg)
    all_times = []
    for packets in eval_episodes:
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
    return {"avg_delivery_time": float(np.mean(all_times))}


# ---------------------------------------------------------------------------
# Train + eval one variant (receives pre-generated episodes)
# ---------------------------------------------------------------------------

def run_variant(label: str, variant_key: str, topo, cfg: ScaIRConfig,
                train_eps: List, eval_eps: List) -> dict:
    # Only reset torch seed — packet randomness is already fixed via pre-generation
    torch.manual_seed(SEED)

    if variant_key == "nbrmask":
        agents = build_agents_no_gnn(topo, cfg, NeighborMaskSubGNN)
    else:
        agents = build_agents(topo, cfg, gnn_cls=PaperSubGNN)

    env = RoutingEnvironment(topo, cfg)
    train_curve: List[float] = []
    t_start = time.time()

    for ep, packets in enumerate(train_eps, start=1):
        stats = env.run_episode(packets, agents, training=True)
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
                  f"sigma={agents[0].sigma:.2f}  ({elapsed:.0f}s)", flush=True)

    eval_times: List[float] = []
    for packets in eval_eps:
        stats = env.run_episode(packets, agents, training=False)
        eval_times.append(stats["avg_delivery_time"])

    return {
        "avg_delivery_time": float(np.mean(eval_times)),
        "std_delivery_time": float(np.std(eval_times)),
        "train_curve": train_curve,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def smooth(values: List[float], w: int = 15) -> List[float]:
    out = []
    for i in range(len(values)):
        lo = max(0, i - w // 2)
        hi = min(len(values), i + w // 2 + 1)
        out.append(float(np.mean(values[lo:hi])))
    return out


def plot_dr_sweep(results: dict, out_path: str) -> None:
    drs = sorted(results.keys())
    ospf = [results[d]["ospf"]["avg_delivery_time"] for d in drs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("NbrMask (no GNN) vs Paper GNN — UCB, Per-node, Abilene", fontsize=13)

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
    print(f"  Saved: {out_path}", flush=True)


def plot_training_curves(results: dict, out_path: str) -> None:
    drs = sorted(results.keys())
    n = len(drs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    fig.suptitle("Training Convergence — Abilene (same episodes per variant)", fontsize=13)
    if n == 1:
        axes = [axes]

    for i, (dr, ax) in enumerate(zip(drs, axes)):
        for key, label, color, _ in VARIANTS:
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
    print(f"  Saved: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)

    topo    = load_topology(TOPO_FILE)
    raw_tms = load_all_traffic_matrices(TM_DIR, topo.num_nodes)
    tms     = [normalise_tm(tm) for tm in raw_tms]
    print(f"Abilene: {topo.num_nodes} nodes, {len(tms)} TMs", flush=True)

    base_cfg = ScaIRConfig()
    max_deg  = max(len(v) for v in topo.adjacency.values())
    if topo.num_nodes > base_cfg.max_nodes:  base_cfg.max_nodes  = topo.num_nodes
    if max_deg        > base_cfg.max_degree: base_cfg.max_degree = max_deg
    base_cfg.action_method = "ucb"

    all_results: dict = {}

    for dr in DR_VALUES:
        print(f"\n{'='*65}")
        print(f"  Abilene  D_r = {dr}  ({TRAIN_EPISODES} train, {EVAL_EPISODES} eval, {N_PACKETS} pkts/ep)")
        print(f"{'='*65}", flush=True)

        cfg = copy.copy(base_cfg)
        cfg.distribution_ratio  = dr
        cfg.packets_per_episode = N_PACKETS

        # Pre-generate ALL episodes once from fixed seed using correct DR config
        env_for_gen = RoutingEnvironment(topo, cfg)
        print("  Pre-generating episodes (fixed seed)...", flush=True)
        all_eps = pregenerate_episodes(env_for_gen, tms,
                                       TRAIN_EPISODES, EVAL_EPISODES,
                                       N_PACKETS, SEED)
        train_eps = all_eps[:TRAIN_EPISODES]
        eval_eps  = all_eps[TRAIN_EPISODES:]

        ospf = run_ospf(topo, cfg, eval_eps)
        print(f"  OSPF: {ospf['avg_delivery_time']:.3f} ms", flush=True)

        dr_results: dict = {"ospf": ospf}
        for key, label, _, _ in VARIANTS:
            print(f"  Running {label}...", flush=True)
            res = run_variant(f"Abilene D_r={dr} {label}", key, topo, cfg,
                              train_eps, eval_eps)
            dr_results[key] = res
            gain = (ospf["avg_delivery_time"] - res["avg_delivery_time"]) / max(ospf["avg_delivery_time"], 1e-6) * 100
            print(f"    {label}: {res['avg_delivery_time']:.3f} ms  ({gain:+.1f}% vs OSPF)", flush=True)

        all_results[dr] = dr_results

    plot_dr_sweep(all_results, os.path.join(RESULTS, "abilene_dr_sweep.png"))
    plot_training_curves(all_results, os.path.join(RESULTS, "abilene_training_curves.png"))

    print(f"\n{'='*70}")
    print("  Abilene summary")
    print(f"{'='*70}")
    for dr in DR_VALUES:
        r = all_results[dr]
        o = r["ospf"]["avg_delivery_time"]
        row = f"  D_r={dr:.1f}  OSPF={o:.3f}"
        for vkey, vlabel, _, _ in VARIANTS:
            t = r[vkey]["avg_delivery_time"]
            g = (o - t) / max(o, 1e-6) * 100
            row += f"  |  {vlabel}: {t:.3f} ({g:+.1f}%)"
        print(row, flush=True)

    json_out: dict = {}
    for dr, res in all_results.items():
        json_out[str(dr)] = {
            "ospf": res["ospf"],
            **{k: {"avg_delivery_time": res[k]["avg_delivery_time"],
                   "std_delivery_time": res[k]["std_delivery_time"]}
               for k, *_ in VARIANTS},
        }
    json_path = os.path.join(RESULTS, "results.json")
    with open(json_path, "w") as f:
        json.dump({"config": {"topology": "abilene", "dr_values": DR_VALUES,
                               "train_episodes": TRAIN_EPISODES,
                               "eval_episodes": EVAL_EPISODES,
                               "n_packets": N_PACKETS, "seed": SEED,
                               "action_method": "ucb",
                               "note": "episodes pre-generated once; both variants see identical traffic"},
                   "results": json_out}, f, indent=2)
    print(f"\nSaved: {json_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
