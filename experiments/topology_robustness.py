#!/usr/bin/env python3
"""
Topology robustness experiment for ScaIR (scair package, fixed 1ms transmission).

Tests how a pretrained ScaIR model handles two topology changes mid-deployment:
  1. ADD NODE: a new node 11 is added, connected to existing nodes 0 and 8.
  2. REMOVE LINK: link 3-6 is cut (simulating a link failure).

Protocol for each change:
  Phase A  (pre-change)   : eval 100 eps, training=False  → steady-state baseline
  Phase B  (immediate)    : apply change, eval 50 eps, training=False  → impact
  Phase C  (adaptation)   : eval 200 eps, training=True   → online adaptation
  Phase D  (post-adapt)   : eval 100 eps, training=False  → recovered performance

OSPF (hop-count Dijkstra) is recomputed instantly on the new topology for reference.

Outputs
-------
  results/topology_robustness_timeline.png   – delivery time across all phases
  results/topology_robustness_summary.png    – bar chart: per-phase summary
  results/topology_robustness.json           – raw numbers
"""

import copy
import heapq
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scair.agent import IRrAgent
from scair.config import ScaIRConfig
from scair.data_loader import load_all_traffic_matrices, load_topology, normalise_tm
from scair.environment import RoutingEnvironment, Topology
from train import build_agents, load_checkpoint


# ---------------------------------------------------------------------------
# OSPF baseline (hop-count Dijkstra)
# ---------------------------------------------------------------------------

def dijkstra_hops(topo: Topology, source: int) -> Dict[int, int]:
    dist = {n: float("inf") for n in range(topo.num_nodes)}
    prev: Dict[int, Optional[int]] = {n: None for n in range(topo.num_nodes)}
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
        while prev[node] != source:
            node = prev[node]
        next_hop[dst] = node
    return next_hop


def run_ospf_episode(topo: Topology, cfg: ScaIRConfig, tm: np.ndarray, n_packets: int) -> float:
    next_hops = {n: dijkstra_hops(topo, n) for n in range(topo.num_nodes)}
    env = RoutingEnvironment(topo, cfg)
    packets = env.generate_packets(tm, n_packets)
    env.reset()
    heap: List = []
    for pkt in packets:
        heapq.heappush(heap, (pkt.birth_time, pkt.pid, pkt, pkt.source, None))
    delivery_times: List[float] = []
    hop_counts: Dict[int, int] = defaultdict(int)
    queue_len: Dict[Tuple[int, int], int] = defaultdict(int)
    while heap:
        t, _, pkt, node, prev = heapq.heappop(heap)
        if prev is not None:
            queue_len[(prev, node)] = max(0, queue_len[(prev, node)] - 1)
        if node == pkt.destination:
            delivery_times.append(t - pkt.birth_time)
            continue
        if hop_counts[pkt.pid] >= cfg.max_hops:
            continue
        next_node = next_hops.get(node, {}).get(pkt.destination)
        if next_node is None:
            continue
        q = queue_len[(node, next_node)]
        cost = q * cfg.queue_time_per_packet + cfg.transmission_time
        queue_len[(node, next_node)] += 1
        hop_counts[pkt.pid] += 1
        heapq.heappush(heap, (t + cost, pkt.pid, pkt, next_node, node))
    return float(np.mean(delivery_times)) if delivery_times else float("inf")


# ---------------------------------------------------------------------------
# Topology mutation helpers
# ---------------------------------------------------------------------------

def add_node(topo: Topology, agents: List[IRrAgent], cfg: ScaIRConfig,
             new_id: int, connect_to: List[int]) -> IRrAgent:
    """
    Add a new node to the topology and create a fresh agent for it.
    Existing agents that gain a new neighbour have their data structures updated;
    their Q-network keeps trained weights (the new-neighbour output slot is random
    and will adapt online).
    """
    topo.num_nodes = new_id + 1
    topo.adjacency[new_id] = list(connect_to)
    for existing in connect_to:
        topo.adjacency[existing] = topo.adjacency[existing] + [new_id]

    new_agent = IRrAgent(new_id, list(connect_to), topo.num_nodes, cfg)
    agents.append(new_agent)

    for existing in connect_to:
        ag = agents[existing]
        if new_id not in ag.neighbours:
            ag.neighbours = ag.neighbours + [new_id]
            ag.degree = len(ag.neighbours)
            ag._nbr_to_idx = {n: i for i, n in enumerate(ag.neighbours)}
            ag.queue_lengths[new_id] = 0

    return new_agent


def remove_link(topo: Topology, agents: List[IRrAgent], cfg: ScaIRConfig,
                u: int, v: int) -> None:
    """
    Remove the link u-v from the topology.
    Affected agents (u and v) are rebuilt with fresh Q-networks but keep the
    same sub_gnn. This simulates a router reboot after a link change.
    """
    topo.adjacency[u] = [x for x in topo.adjacency[u] if x != v]
    topo.adjacency[v] = [x for x in topo.adjacency[v] if x != u]

    for node in (u, v):
        old = agents[node]
        new_ag = IRrAgent(node, topo.adjacency[node], topo.num_nodes, cfg)
        new_ag.sub_gnn = old.sub_gnn        # keep learned topology representation
        new_ag._owns_gnn = old._owns_gnn
        new_ag.tick = old.tick
        new_ag.sigma = old.sigma
        agents[node] = new_ag


# ---------------------------------------------------------------------------
# Run a phase: N episodes, return per-episode delivery times
# ---------------------------------------------------------------------------

def run_phase(topo: Topology, cfg: ScaIRConfig, tms: List[np.ndarray],
              agents: List[IRrAgent], n_episodes: int,
              n_packets: int, training: bool) -> List[float]:
    env = RoutingEnvironment(topo, cfg)
    times = []
    for ep in range(n_episodes):
        tm = tms[ep % len(tms)]
        pkts = env.generate_packets(tm, n_packets)
        stats = env.run_episode(pkts, agents, training=training)
        times.append(stats["avg_delivery_time"])
    return times


def run_ospf_phase(topo: Topology, cfg: ScaIRConfig, tms: List[np.ndarray],
                   n_episodes: int, n_packets: int) -> List[float]:
    return [run_ospf_episode(topo, cfg, tms[ep % len(tms)], n_packets)
            for ep in range(n_episodes)]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

COLORS = {
    "scair": "tab:blue",
    "ospf":  "tab:orange",
}

PHASE_COLORS = ["#e8f4f8", "#fdebd0", "#d5f5e3", "#eaf2ff"]


def plot_timeline(all_scair: List[float], all_ospf: List[float],
                  phase_labels: List[str], phase_lengths: List[int],
                  title: str, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))

    x = list(range(len(all_scair)))
    smooth_scair = np.convolve(all_scair, np.ones(5) / 5, mode="same")
    smooth_ospf  = np.convolve(all_ospf,  np.ones(5) / 5, mode="same")

    # Phase background bands
    start = 0
    for i, (label, length) in enumerate(zip(phase_labels, phase_lengths)):
        ax.axvspan(start, start + length, alpha=0.25, color=PHASE_COLORS[i % 4], label=label)
        ax.text(start + length / 2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                label, ha="center", va="top", fontsize=7, color="gray")
        if i > 0:
            ax.axvline(start, color="gray", linestyle="--", linewidth=0.8)
        start += length

    ax.plot(x, all_scair, alpha=0.3, color=COLORS["scair"], linewidth=0.8)
    ax.plot(x, smooth_scair, color=COLORS["scair"], linewidth=2, label="ScaIR")
    ax.plot(x, all_ospf,  alpha=0.3, color=COLORS["ospf"],  linewidth=0.8)
    ax.plot(x, smooth_ospf,  color=COLORS["ospf"],  linewidth=2, label="OSPF (hop)", linestyle="--")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg delivery time (ms)")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_summary(results: dict, save_path: str) -> None:
    experiments = list(results.keys())
    phase_names = ["A: Pre-change", "B: Immediate", "C: Adaptation", "D: Recovered"]

    n_exp = len(experiments)
    n_phases = len(phase_names)
    x = np.arange(n_phases)
    width = 0.35

    fig, axes = plt.subplots(1, n_exp, figsize=(6 * n_exp, 5), sharey=False)
    if n_exp == 1:
        axes = [axes]

    for ax, exp_name in zip(axes, experiments):
        data = results[exp_name]
        scair_means = [np.mean(data["scair"][p]) for p in phase_names]
        ospf_means  = [np.mean(data["ospf"][p])  for p in phase_names]

        bars1 = ax.bar(x - width / 2, scair_means, width, label="ScaIR",
                       color=COLORS["scair"], alpha=0.8)
        bars2 = ax.bar(x + width / 2, ospf_means,  width, label="OSPF",
                       color=COLORS["ospf"],  alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(phase_names, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("Avg delivery time (ms)")
        ax.set_title(exp_name, fontsize=11)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Topology Robustness: ScaIR vs OSPF per Phase", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    TOPO_PATH  = "data/ABI/Topology.txt"
    TM_DIR     = "data/ABI/TrafficMatrix"
    CHECKPOINT = "checkpoints/ABI/episode_99750.pt"
    N_PACKETS  = 100
    D_R        = 0.7

    PHASES = {
        "A: Pre-change":  {"eps": 100, "train": False},
        "B: Immediate":   {"eps":  50, "train": False},
        "C: Adaptation":  {"eps": 200, "train": True},
        "D: Recovered":   {"eps": 100, "train": False},
    }

    os.makedirs("results", exist_ok=True)

    cfg = ScaIRConfig(
        packets_per_episode=N_PACKETS,
        distribution_ratio=D_R,
        sigma_initial=0.1,
        sigma_min=0.1,
    )

    topo_orig = load_topology(TOPO_PATH)
    raw_tms = load_all_traffic_matrices(TM_DIR, topo_orig.num_nodes)
    tms = [normalise_tm(tm) for tm in raw_tms]

    max_deg = max(len(v) for v in topo_orig.adjacency.values())
    if topo_orig.num_nodes > cfg.max_nodes: cfg.max_nodes = topo_orig.num_nodes
    if max_deg > cfg.max_degree: cfg.max_degree = max_deg

    results = {}
    all_timelines = {}

    for exp_name, (change_fn, change_kwargs) in [
        ("Add Node 11 (→0, →8)", (add_node,    {"new_id": 11, "connect_to": [0, 8]})),
        ("Remove Link 3-6",       (remove_link, {"u": 3, "v": 6})),
    ]:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print('='*60)

        # Fresh copies for this experiment
        topo = copy.deepcopy(topo_orig)
        agents = build_agents(topo, cfg)
        load_checkpoint(agents, CHECKPOINT)

        phase_scair: Dict[str, List[float]] = {}
        phase_ospf:  Dict[str, List[float]] = {}
        timeline_scair: List[float] = []
        timeline_ospf:  List[float] = []
        phase_lengths = []

        for i, (phase_name, phase_cfg) in enumerate(PHASES.items()):
            # Apply topology change right before Phase B
            if i == 1:
                print(f"  >> Applying topology change: {exp_name}")
                change_fn(topo, agents, cfg, **change_kwargs)
                # Expand TMs for new node if needed
                if topo.num_nodes > len(tms[0]):
                    tms = [np.pad(tm, ((0, topo.num_nodes - len(tm)),
                                       (0, topo.num_nodes - len(tm)))) for tm in tms]
                if topo.num_nodes > cfg.max_nodes:
                    cfg.max_nodes = topo.num_nodes

            print(f"  Phase {phase_name}: {phase_cfg['eps']} eps, training={phase_cfg['train']}")
            sc = run_phase(topo, cfg, tms, agents, phase_cfg["eps"], N_PACKETS, phase_cfg["train"])
            os_ = run_ospf_phase(topo, cfg, tms, phase_cfg["eps"], N_PACKETS)

            phase_scair[phase_name] = sc
            phase_ospf[phase_name]  = os_
            timeline_scair.extend(sc)
            timeline_ospf.extend(os_)
            phase_lengths.append(phase_cfg["eps"])

            print(f"    ScaIR: {np.mean(sc):.2f} ms  |  OSPF: {np.mean(os_):.2f} ms")

        results[exp_name] = {"scair": phase_scair, "ospf": phase_ospf}
        all_timelines[exp_name] = {
            "scair": timeline_scair, "ospf": timeline_ospf,
            "phase_labels": list(PHASES.keys()), "phase_lengths": phase_lengths
        }

    # --- Plots ---
    print("\nGenerating plots...")

    fig, axes = plt.subplots(len(all_timelines), 1,
                              figsize=(14, 5 * len(all_timelines)))
    if len(all_timelines) == 1:
        axes = [axes]

    phase_names_list = list(PHASES.keys())
    phase_lengths_list = [PHASES[p]["eps"] for p in phase_names_list]

    for ax, (exp_name, tl) in zip(axes, all_timelines.items()):
        sc = tl["scair"]; os_ = tl["ospf"]
        x = range(len(sc))
        k = 7  # smoothing window

        def smooth(v): return np.convolve(v, np.ones(k) / k, mode="same")

        # Phase bands
        start = 0
        for i, (label, length) in enumerate(zip(phase_names_list, phase_lengths_list)):
            ax.axvspan(start, start + length, alpha=0.18,
                       color=PHASE_COLORS[i], label=label)
            ax.text(start + length / 2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                    label, ha="center", va="top", fontsize=7, color="#555")
            if i > 0:
                ax.axvline(start, color="gray", linestyle="--", linewidth=0.8)
            start += length

        ax.plot(x, sc,  alpha=0.2, color=COLORS["scair"], linewidth=0.6)
        ax.plot(x, smooth(sc), color=COLORS["scair"], linewidth=2, label="ScaIR")
        ax.plot(x, os_, alpha=0.2, color=COLORS["ospf"],  linewidth=0.6)
        ax.plot(x, smooth(os_), color=COLORS["ospf"],  linewidth=2, label="OSPF", linestyle="--")

        ax.set_title(exp_name, fontsize=11)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg delivery time (ms)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Topology Robustness  |  D_r={D_R}, {N_PACKETS} pkts/ep", fontsize=13)
    plt.tight_layout()
    timeline_path = "results/topology_robustness_timeline.png"
    plt.savefig(timeline_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {timeline_path}")

    plot_summary(results, "results/topology_robustness_summary.png")

    # Save JSON
    json_results = {}
    for exp, data in results.items():
        json_results[exp] = {
            "scair": {p: float(np.mean(v)) for p, v in data["scair"].items()},
            "ospf":  {p: float(np.mean(v)) for p, v in data["ospf"].items()},
        }
    with open("results/topology_robustness.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print("  Saved → results/topology_robustness.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
