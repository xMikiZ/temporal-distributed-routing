#!/usr/bin/env python3
"""
Topology robustness experiment for all ScaIR variants.

Each variant is trained from scratch on the base Abilene topology, then
subjected to four topology mutations one at a time:

  1. Add node  -- new router 11 connected to nodes 0 and 8
  2. Remove link -- cut link between nodes 3 and 6
  3. Add link    -- new shortcut between nodes 1 and 7
  4. Remove node -- node 5 and all its links disappear

Protocol per mutation (same for every variant):
  Phase A: Evaluate pre-change       (EVAL_EPS episodes)
  [topology change applied]
  Phase B: Evaluate immediately      (EVAL_EPS episodes, no adaptation)
  Phase C: Online adaptation         (ADAPT_EPS training episodes)
  Phase D: Evaluate post-adaptation  (EVAL_EPS episodes)

Outputs
-------
  results/02_topology_robustness/robustness_summary.png
  results/02_topology_robustness/robustness_results.json
"""

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scair.config import ScaIRConfig
from scair.data_loader import load_all_traffic_matrices, load_topology, normalise_tm
from scair.environment import RoutingEnvironment
from scair.models import AttentionSubGNN
from train import build_agents, build_agents_shared_gnn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOPO_FILE    = "data/ABI/Topology.txt"
TM_DIR       = "data/ABI/TrafficMatrix"
RESULTS      = "results/02_topology_robustness"
SEED         = 42

TRAIN_EPS    = 300    # initial training on base topology
EVAL_EPS     = 50     # evaluation phases A, B, D
ADAPT_EPS    = 200    # phase C online adaptation
N_PACKETS    = 100
D_R          = 0.4    # moderate hot-spot pressure

VARIANTS = [
    ("per_node",      "Per-node SubGNN",    "steelblue",    False, None),
    ("shared",        "Shared SubGNN",      "darkorange",   True,  None),
    ("attn_per_node", "Per-node Attention", "mediumorchid", False, AttentionSubGNN),
    ("attn_shared",   "Shared Attention",   "crimson",      True,  AttentionSubGNN),
]

MUTATIONS = [
    ("add_node",    "Add node 11\n(connect→0,8)"),
    ("remove_link", "Remove link\n3–6"),
    ("add_link",    "Add link\n1–7"),
    ("remove_node", "Remove node 5\n(+ its links)"),
]


# ---------------------------------------------------------------------------
# Topology mutation helpers
# ---------------------------------------------------------------------------

def apply_mutation(topo, mutation: str):
    """Return a (possibly modified) copy of topo after the given mutation."""
    t = copy.deepcopy(topo)
    if mutation == "add_node":
        new_id = t.num_nodes          # node 11
        t.num_nodes += 1
        t.adjacency[new_id] = [0, 8]
        t.adjacency[0].append(new_id)
        t.adjacency[8].append(new_id)
    elif mutation == "remove_link":
        if 6 in t.adjacency.get(3, []):
            t.adjacency[3].remove(6)
        if 3 in t.adjacency.get(6, []):
            t.adjacency[6].remove(3)
    elif mutation == "add_link":
        if 7 not in t.adjacency.get(1, []):
            t.adjacency[1].append(7)
        if 1 not in t.adjacency.get(7, []):
            t.adjacency[7].append(1)
    elif mutation == "remove_node":
        node = 5
        nbrs = list(t.adjacency.get(node, []))
        for nb in nbrs:
            if node in t.adjacency.get(nb, []):
                t.adjacency[nb].remove(node)
        t.adjacency[node] = []  # keep key so range(num_nodes) doesn't KeyError
    return t


# ---------------------------------------------------------------------------
# Agent adaptation helpers
# ---------------------------------------------------------------------------

def adapt_agents_to_topo(agents, old_topo, new_topo, cfg, is_shared, gnn_cls):
    """
    Update agents in-place to handle topology changes:
    - Add node: append a new agent
    - Remove link: rebuild the affected agents' neighbour lists
    - Add link: rebuild the affected agents' neighbour lists
    - Remove node: mark node 5's agent as inactive (it just drops packets)
    Returns the (possibly extended) agents list and a new cfg.
    """
    new_cfg = copy.copy(cfg)
    max_deg = max(len(v) for v in new_topo.adjacency.values())
    if new_topo.num_nodes > new_cfg.max_nodes:
        new_cfg.max_nodes = new_topo.num_nodes
    if max_deg > new_cfg.max_degree:
        new_cfg.max_degree = max_deg
    # We keep the same agents list; neighbour lists are re-read from topo
    # at routing time via env. For simplicity we return agents as-is.
    return agents, new_cfg


# ---------------------------------------------------------------------------
# OSPF baseline (queue simulation)
# ---------------------------------------------------------------------------

def _dijkstra(topo, source: int) -> Dict[int, int]:
    dist = {n: float("inf") for n in topo.adjacency}
    prev: Dict[int, int] = {}
    dist[source] = 0
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v in topo.adjacency.get(u, []):
            if v not in dist:
                continue
            nd = d + 1
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    next_hop: Dict[int, int] = {}
    for dst in list(topo.adjacency.keys()):
        if dst == source or dist.get(dst, float("inf")) == float("inf"):
            continue
        node = dst
        while prev.get(node, source) != source:
            node = prev[node]
        next_hop[dst] = node
    return next_hop


def run_ospf(topo, cfg, tms, n_eps, n_pkts) -> float:
    valid_nodes = list(topo.adjacency.keys())
    routes = {n: _dijkstra(topo, n) for n in valid_nodes}
    env = RoutingEnvironment(topo, cfg)
    times = []
    for ep in range(n_eps):
        tm = tms[ep % len(tms)]
        packets = env.generate_packets(tm, n_pkts)
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
        times.append(float(np.mean(delivery_times)) if delivery_times else float("inf"))
    return float(np.mean(times))


# ---------------------------------------------------------------------------
# Core training / evaluation
# ---------------------------------------------------------------------------

def build(variant_key, is_shared, gnn_cls, topo, cfg):
    if is_shared:
        return build_agents_shared_gnn(topo, cfg, gnn_cls=gnn_cls)
    return build_agents(topo, cfg, gnn_cls=gnn_cls)


def train(agents, topo, cfg, tms, n_eps, is_shared, label=""):
    env = RoutingEnvironment(topo, cfg)
    curve = []
    for ep in range(1, n_eps + 1):
        tm = tms[(ep - 1) % len(tms)]
        pkts = env.generate_packets(tm, cfg.packets_per_episode)
        stats = env.run_episode(pkts, agents, training=True)
        if is_shared:
            agents[0].shared_gnn_step(len(agents))
        curve.append(stats["avg_delivery_time"])
        if ep == 10:
            for ag in agents: ag.set_learning_rate(cfg.learning_rate)
        if ep % cfg.target_update_freq == 0:
            for ag in agents: ag.update_target()
        if ep % cfg.sigma_decay_freq == 0:
            for ag in agents: ag.decay_sigma()
    return curve


def evaluate(agents, topo, cfg, tms, n_eps, offset=0) -> float:
    env = RoutingEnvironment(topo, cfg)
    times = []
    for ep in range(1, n_eps + 1):
        tm = tms[(offset + ep - 1) % len(tms)]
        pkts = env.generate_packets(tm, cfg.packets_per_episode)
        stats = env.run_episode(pkts, agents, training=False)
        times.append(stats["avg_delivery_time"])
    return float(np.mean(times))


# ---------------------------------------------------------------------------
# Run one variant through all mutations
# ---------------------------------------------------------------------------

def _add_node_agent(new_id, new_topo, new_cfg, is_shared, gnn_cls, agents):
    """Create a new agent for node new_id after an add_node mutation.

    For shared variants: the new agent reuses f_w/g_w from agents[0] via
    make_shared_node_gnn so weights stay shared.  For per-node variants a
    fresh IRrAgent is created with its own SubGNN.
    """
    from scair.agent import IRrAgent
    nbrs = new_topo.adjacency[new_id]
    if is_shared:
        from scair.models import make_shared_node_gnn
        # agents[0].sub_gnn already has the right f_w/g_w (shared tensors)
        new_gnn = make_shared_node_gnn(new_id, new_cfg.feature_length, agents[0].sub_gnn)
        return IRrAgent(
            node_id=new_id, neighbours=nbrs,
            num_nodes=new_topo.num_nodes, cfg=new_cfg,
            shared_sub_gnn=new_gnn, shared_gnn_opt=None,
        )
    return IRrAgent(
        node_id=new_id, neighbours=nbrs,
        num_nodes=new_topo.num_nodes, cfg=new_cfg,
        gnn_cls=gnn_cls,
    )


def run_variant(vkey, vname, color, is_shared, gnn_cls, base_topo, tms, base_cfg):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print(f"\n  [{vname}] Training on base topology ({TRAIN_EPS} eps)...")
    agents = build(vkey, is_shared, gnn_cls, base_topo, base_cfg)
    train(agents, base_topo, base_cfg, tms, TRAIN_EPS, is_shared)

    mutation_results = {}
    for mut_key, mut_label in MUTATIONS:
        print(f"    Mutation: {mut_key}")

        # Phase A: baseline on original topology
        t_A = evaluate(agents, base_topo, base_cfg, tms, EVAL_EPS, offset=TRAIN_EPS)

        # Apply mutation
        new_topo = apply_mutation(base_topo, mut_key)
        new_cfg = copy.copy(base_cfg)
        max_deg = max(len(v) for v in new_topo.adjacency.values() if v)
        if new_topo.num_nodes > new_cfg.max_nodes:
            new_cfg.max_nodes = new_topo.num_nodes
        if max_deg > new_cfg.max_degree:
            new_cfg.max_degree = max_deg

        # For add_node: create an agent for the new node before env calls agents[new_id]
        eval_agents = list(agents)
        if mut_key == "add_node":
            new_id = base_topo.num_nodes  # node 11
            eval_agents = agents + [_add_node_agent(new_id, new_topo, new_cfg,
                                                    is_shared, gnn_cls, agents)]

        # OSPF on new topology
        t_ospf = run_ospf(new_topo, new_cfg, tms, EVAL_EPS, N_PACKETS)

        # Phase B: immediate eval on new topology (no adaptation)
        t_B = evaluate(eval_agents, new_topo, new_cfg, tms, EVAL_EPS, offset=TRAIN_EPS)

        # Phase C: adaptation
        adapt_curve = train(eval_agents, new_topo, new_cfg, tms, ADAPT_EPS, is_shared)

        # Phase D: post-adaptation eval
        t_D = evaluate(eval_agents, new_topo, new_cfg, tms, EVAL_EPS,
                       offset=TRAIN_EPS + ADAPT_EPS)

        # Degradation: how much worse immediately after change (vs pre-change)
        # Recovery: what fraction of the degradation was recovered by adaptation
        degrade = (t_B - t_A) / max(t_A, 1e-6) * 100
        recover = (t_B - t_D) / max(abs(t_B - t_A) + 1e-6, 1e-6) * 100

        mutation_results[mut_key] = {
            "phase_A": t_A, "phase_B": t_B, "phase_D": t_D,
            "ospf_new": t_ospf,
            "adapt_curve": adapt_curve,
            "degradation_pct": degrade,
            "recovery_pct": recover,
        }
        print(f"      A={t_A:.2f}ms  B={t_B:.2f}ms  D={t_D:.2f}ms  "
              f"OSPF={t_ospf:.2f}ms  degrade={degrade:+.1f}%  recover={recover:.1f}%")

        # Reset agents to trained base-topology state for the next mutation
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        agents = build(vkey, is_shared, gnn_cls, base_topo, base_cfg)
        train(agents, base_topo, base_cfg, tms, TRAIN_EPS, is_shared)

    return mutation_results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def smooth(vals, w=10):
    out = []
    for i in range(len(vals)):
        lo, hi = max(0, i - w // 2), min(len(vals), i + w // 2 + 1)
        out.append(float(np.mean(vals[lo:hi])))
    return out


def plot_summary(all_results, out_path):
    mutations = [m[0] for m in MUTATIONS]
    mut_labels = [m[1] for m in MUTATIONS]
    variants = [(vk, vn, vc) for vk, vn, vc, *_ in VARIANTS]
    phases = ["phase_A", "phase_B", "phase_D"]
    phase_labels = ["Pre-change\n(Phase A)", "Post-change\n(Phase B)", "Adapted\n(Phase D)"]

    n_mut = len(mutations)
    fig, axes = plt.subplots(1, n_mut, figsize=(5 * n_mut, 5), sharey=False)
    fig.suptitle(
        "Topology Robustness: ScaIR Variants\n"
        f"(Abilene, D_r={D_R}, {TRAIN_EPS} base train eps, {ADAPT_EPS} adaptation eps)",
        fontsize=12,
    )

    x = np.arange(len(phases))
    width = 0.18

    for ax, mut, mlabel in zip(axes, mutations, mut_labels):
        for i, (vk, vn, vc) in enumerate(variants):
            vals = [all_results[vk][mut][p] for p in phases]
            bars = ax.bar(x + (i - 1.5) * width, vals, width, label=vn, color=vc, alpha=0.85)
        # OSPF reference line
        ospf_val = list(all_results.values())[0][mut]["ospf_new"]
        ax.axhline(ospf_val, color="green", linestyle="--", linewidth=1.5, label="OSPF (new topo)")
        ax.set_title(mlabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(phase_labels, fontsize=8)
        ax.set_ylabel("Avg Delivery Time (ms)")
        ax.grid(axis="y", alpha=0.3)

    axes[-1].legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_adaptation_curves(all_results, out_path):
    mutations = [m[0] for m in MUTATIONS]
    mut_labels = [m[1] for m in MUTATIONS]

    n_mut = len(mutations)
    fig, axes = plt.subplots(1, n_mut, figsize=(5 * n_mut, 4), sharey=False)
    fig.suptitle("Adaptation Curves After Topology Change", fontsize=12)

    for ax, mut, mlabel in zip(axes, mutations, mut_labels):
        for vk, vn, vc, *_ in VARIANTS:
            curve = smooth(all_results[vk][mut]["adapt_curve"])
            ax.plot(range(1, len(curve) + 1), curve, color=vc, linewidth=1.5, label=vn)
        ospf_val = list(all_results.values())[0][mut]["ospf_new"]
        ax.axhline(ospf_val, color="green", linestyle="--", linewidth=1.2, label="OSPF (new topo)")
        ax.set_title(mlabel, fontsize=10)
        ax.set_xlabel("Adaptation episode")
        ax.set_ylabel("Avg Delivery Time (ms)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS, exist_ok=True)

    print("Loading topology and traffic matrices...")
    base_topo = load_topology(TOPO_FILE)
    raw_tms   = load_all_traffic_matrices(TM_DIR, base_topo.num_nodes)
    tms       = [normalise_tm(tm) for tm in raw_tms]
    print(f"  {base_topo.num_nodes} nodes, {len(tms)} TMs")

    base_cfg = ScaIRConfig()
    max_deg  = max(len(v) for v in base_topo.adjacency.values())
    if base_topo.num_nodes > base_cfg.max_nodes:  base_cfg.max_nodes  = base_topo.num_nodes
    if max_deg             > base_cfg.max_degree: base_cfg.max_degree = max_deg
    base_cfg.distribution_ratio  = D_R
    base_cfg.packets_per_episode = N_PACKETS

    all_results = {}
    for vkey, vname, color, is_shared, gnn_cls in VARIANTS:
        print(f"\n{'='*60}\n  Variant: {vname}\n{'='*60}")
        all_results[vkey] = run_variant(
            vkey, vname, color, is_shared, gnn_cls, base_topo, tms, base_cfg
        )

    print("\nGenerating plots...")
    plot_summary(all_results, os.path.join(RESULTS, "robustness_summary.png"))
    plot_adaptation_curves(all_results, os.path.join(RESULTS, "adaptation_curves.png"))

    # Save JSON
    json_out = {}
    for vkey, res in all_results.items():
        json_out[vkey] = {}
        for mut, mres in res.items():
            json_out[vkey][mut] = {k: v for k, v in mres.items() if k != "adapt_curve"}
    with open(os.path.join(RESULTS, "robustness_results.json"), "w") as f:
        json.dump({"config": {"train_eps": TRAIN_EPS, "eval_eps": EVAL_EPS,
                               "adapt_eps": ADAPT_EPS, "d_r": D_R, "seed": SEED},
                   "results": json_out}, f, indent=2)
    print(f"  Saved: {os.path.join(RESULTS, 'robustness_results.json')}")

    # Summary table
    print("\n" + "="*80)
    print(f"{'Variant':<20}  {'Mutation':<15}  {'A':>7}  {'B':>7}  {'D':>7}  {'Degrade%':>9}  {'Recover%':>9}")
    print("="*80)
    for vkey, vname, *_ in VARIANTS:
        for mut_key, _ in MUTATIONS:
            r = all_results[vkey][mut_key]
            print(f"  {vname:<18}  {mut_key:<15}  {r['phase_A']:7.2f}  "
                  f"{r['phase_B']:7.2f}  {r['phase_D']:7.2f}  "
                  f"{r['degradation_pct']:+8.1f}%  {r['recovery_pct']:+8.1f}%")
    print("="*80)


if __name__ == "__main__":
    main()
