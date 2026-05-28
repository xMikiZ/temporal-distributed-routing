#!/usr/bin/env python3
"""
Paper f_w vs Ours on Germany50.

'Ours' results are loaded from the existing Exp 4 run (300 train eps, per_node UCB).
Only the paper variant (mean_y f_w(V_y)) is re-trained here, for 200 episodes.

Outputs
-------
  results/07_paper_vs_ours/ger50_dr_sweep.png
  results/07_paper_vs_ours/ger50_training_curves.png
  results/07_paper_vs_ours/ger50_results.json
  checkpoints/ger50_paper/<dr>/episode_0200.pt
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
from scair.models import PaperSubGNN
from train import build_agents

RESULTS        = "results/07_paper_vs_ours"
CHECKPOINTS    = "checkpoints/ger50_paper"
SEED           = 42
DR_VALUES      = [0.0, 0.2, 0.4, 0.6, 0.8]
TRAIN_EPISODES = 200
EVAL_EPISODES  = 50
N_PACKETS      = 100
LOG_INTERVAL   = 50

TOPO_FILE = "data/GER50/Topology.txt"
TM_DIR    = "data/GER50/TrafficMatrix"

# Existing Exp-4 results file (our variant, 300 train eps)
EXISTING_RESULTS = "results/04_germany50_ucb/comparison_experiment.json"

VARIANTS = [
    ("ours",  "Ours (concat V_own+nbr)", "steelblue", "o"),
    ("paper", "Paper (mean f_w(V_y))",   "crimson",   "s"),
]


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


def run_ospf(topo, cfg: ScaIRConfig, tms, n_episodes: int, n_packets: int) -> dict:
    routes = {n: _dijkstra_next_hop(topo, n) for n in range(topo.num_nodes)}
    env = RoutingEnvironment(topo, cfg)
    all_times = []
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
    return {"avg_delivery_time": float(np.mean(all_times))}


# ---------------------------------------------------------------------------
# Train + eval paper variant
# ---------------------------------------------------------------------------

def run_paper_variant(topo, tms, cfg: ScaIRConfig, dr: float) -> dict:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    agents = build_agents(topo, cfg, gnn_cls=PaperSubGNN)
    env = RoutingEnvironment(topo, cfg)

    train_curve: List[float] = []
    t_start = time.time()

    for ep in range(1, TRAIN_EPISODES + 1):
        tm   = tms[(ep - 1) % len(tms)]
        pkts = env.generate_packets(tm, cfg.packets_per_episode)
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
            print(f"  [paper D_r={dr}] ep {ep:3d}/{TRAIN_EPISODES}  "
                  f"time={stats['avg_delivery_time']:6.2f}ms  "
                  f"sigma={agents[0].sigma:.2f}  ({elapsed:.0f}s)", flush=True)

    # Save checkpoint
    ckpt_dir = os.path.join(CHECKPOINTS, f"dr_{dr:.1f}")
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {
        str(n): {
            "sub_gnn": agents[n].sub_gnn.state_dict(),
            "q_net":   agents[n].q_net.state_dict(),
            "sigma":   agents[n].sigma,
        }
        for n in range(len(agents))
    }
    ckpt_path = os.path.join(ckpt_dir, f"episode_{TRAIN_EPISODES:04d}.pt")
    torch.save(state, ckpt_path)
    print(f"  [checkpoint saved to {ckpt_path}]", flush=True)

    # Eval
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
    fig.suptitle("Paper f_w vs Ours — UCB, Per-node, Germany50", fontsize=13)

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
    # Only plot DRs where we have paper train curves
    drs_with_curves = [d for d in drs if results[d]["paper"].get("train_curve")]
    if not drs_with_curves:
        return
    n = len(drs_with_curves)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    fig.suptitle("Training Convergence — Germany50 (paper variant, 200 eps)", fontsize=13)
    if n == 1:
        axes = [axes]

    for i, (dr, ax) in enumerate(zip(drs_with_curves, axes)):
        curve = smooth(results[dr]["paper"]["train_curve"])
        ax.plot(range(1, len(curve) + 1), curve, color="crimson", linewidth=1.5,
                label="Paper (mean f_w(V_y))")
        ax.axhline(results[dr]["ospf"]["avg_delivery_time"], color="green",
                   linestyle="--", linewidth=1.2, label="OSPF")
        ax.axhline(results[dr]["ours"]["avg_delivery_time"], color="steelblue",
                   linestyle=":", linewidth=1.2, label="Ours (eval)")
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
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(RESULTS, exist_ok=True)

    # Load existing 'ours' results from Exp 4
    with open(EXISTING_RESULTS) as f:
        exp4 = json.load(f)
    existing = exp4["results"]  # keyed by str(dr)

    topo    = load_topology(TOPO_FILE)
    raw_tms = load_all_traffic_matrices(TM_DIR, topo.num_nodes)
    tms     = [normalise_tm(tm) for tm in raw_tms]
    print(f"Germany50: {topo.num_nodes} nodes, {len(tms)} TMs", flush=True)

    base_cfg = ScaIRConfig()
    max_deg  = max(len(v) for v in topo.adjacency.values())
    if topo.num_nodes > base_cfg.max_nodes:  base_cfg.max_nodes  = topo.num_nodes
    if max_deg        > base_cfg.max_degree: base_cfg.max_degree = max_deg
    base_cfg.action_method = "ucb"

    topo_results: dict = {}

    for dr in DR_VALUES:
        dr_str = str(dr)
        print(f"\n{'='*65}")
        print(f"  Germany50  D_r = {dr}  ({TRAIN_EPISODES} train eps, {N_PACKETS} pkts/ep)")
        print(f"{'='*65}", flush=True)

        cfg = copy.copy(base_cfg)
        cfg.distribution_ratio  = dr
        cfg.packets_per_episode = N_PACKETS

        # OSPF — reuse from existing results if available, else compute
        if dr_str in existing and "ospf" in existing[dr_str]:
            ospf = {"avg_delivery_time": existing[dr_str]["ospf"]["avg_delivery_time"]}
            print(f"  OSPF (from Exp4): {ospf['avg_delivery_time']:.3f} ms", flush=True)
        else:
            print(f"  OSPF baseline ({EVAL_EPISODES} eps)...", flush=True)
            ospf = run_ospf(topo, cfg, tms, EVAL_EPISODES, N_PACKETS)
            print(f"    OSPF: {ospf['avg_delivery_time']:.3f} ms", flush=True)

        # 'Ours' results from Exp 4 (per_node UCB, 300 train eps)
        ours_data = existing[dr_str]["per_node"]
        ours = {
            "avg_delivery_time": ours_data["avg_delivery_time"],
            "std_delivery_time": ours_data.get("std_delivery_time", 0.0),
            "train_curve": [],  # not available from Exp 4
        }
        gain_ours = (ospf["avg_delivery_time"] - ours["avg_delivery_time"]) / max(ospf["avg_delivery_time"], 1e-6) * 100
        print(f"  Ours (Exp4, 300 eps): {ours['avg_delivery_time']:.3f} ms  ({gain_ours:+.1f}% vs OSPF)", flush=True)

        # Run paper variant
        print(f"  Training paper variant ({TRAIN_EPISODES} eps)...", flush=True)
        paper = run_paper_variant(topo, tms, cfg, dr)
        gain_paper = (ospf["avg_delivery_time"] - paper["avg_delivery_time"]) / max(ospf["avg_delivery_time"], 1e-6) * 100
        print(f"  Paper: {paper['avg_delivery_time']:.3f} ms  ({gain_paper:+.1f}% vs OSPF)", flush=True)

        topo_results[dr] = {"ospf": ospf, "ours": ours, "paper": paper}

    # Plots
    plot_dr_sweep(topo_results,
                  os.path.join(RESULTS, "ger50_dr_sweep.png"))
    plot_training_curves(topo_results,
                         os.path.join(RESULTS, "ger50_training_curves.png"))

    # Summary
    print(f"\n{'='*70}")
    print("  Germany50 summary")
    print(f"{'='*70}")
    for dr in DR_VALUES:
        r = topo_results[dr]
        o = r["ospf"]["avg_delivery_time"]
        row = f"  D_r={dr:.1f}  OSPF={o:.3f}"
        for vkey, vlabel, _, _ in VARIANTS:
            t = r[vkey]["avg_delivery_time"]
            g = (o - t) / max(o, 1e-6) * 100
            row += f"  |  {vlabel}: {t:.3f} ({g:+.1f}%)"
        print(row, flush=True)

    # Save JSON (no train_curve for 'ours', paper curve included)
    json_out: dict = {}
    for dr, res in topo_results.items():
        json_out[str(dr)] = {
            "ospf":  {"avg_delivery_time": res["ospf"]["avg_delivery_time"]},
            "ours":  {"avg_delivery_time": res["ours"]["avg_delivery_time"],
                      "std_delivery_time": res["ours"]["std_delivery_time"],
                      "note": "from Exp4 per_node UCB, 300 train eps"},
            "paper": {"avg_delivery_time": res["paper"]["avg_delivery_time"],
                      "std_delivery_time": res["paper"]["std_delivery_time"]},
        }
    json_path = os.path.join(RESULTS, "ger50_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "config": {
                "topology": "germany50",
                "dr_values": DR_VALUES,
                "paper_train_episodes": TRAIN_EPISODES,
                "ours_train_episodes": 300,
                "eval_episodes": EVAL_EPISODES,
                "n_packets": N_PACKETS,
                "seed": SEED,
                "action_method": "ucb",
            },
            "results": json_out,
        }, f, indent=2)
    print(f"\nSaved: {json_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
