#!/usr/bin/env python3
"""
Experiment 9: Topology-derived GNN initialisation vs fixed encoding.

Three initialisation strategies (degree, betweenness, shortest-path distances)
are tested in two modes:
  - GNN mode: the vector seeds V_n; message-passing refines it via one of
    three aggregation schemes (mean, dot-attention, learnable-attention).
  - Fixed mode: the vector is fed directly as Q-network input (no GNN).

Baseline: ScaIR with one-hot init and concat(V_own, mean(V_nbrs)) aggregation.

Topologies : Abilene, GEANT, BRAIN, Germany50   (UCB, per-node Q-nets)
DR values  : 0.0, 0.4, 0.8  (low / medium / high congestion)
Train eps  : 200
Eval eps   : 50
Packets    : 100

Episodes are pre-generated once per (topology, DR) and shared across all
variants so the comparison is perfectly controlled.

Topology adaptation (Abilene only, DR=0.6):
  Scenario A — add node 11 connected to nodes 0 and 8
  Scenario B — remove link 3-6
  Evaluated immediately (0 adaptation episodes) and after 50 online episodes.

Outputs
-------
  results/09_topo_init/
    {topo_key}_dr_sweep.png
    {topo_key}_results.json
    adaptation_results.json
    adaptation_plot.png
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
from scair.models import (PaperSubGNN, DotAttnSubGNN, LearnableAttnSubGNN,
                           NeighborMaskSubGNN)
from scair.topology_features import compute_init_vectors
from train import (build_agents, build_agents_topo_init,
                   build_agents_fixed_topo, build_agents_no_gnn)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS        = "results/09_topo_init"
SEED           = 42
DR_VALUES      = [0.0, 0.4, 0.8]
TRAIN_EPISODES = 200
EVAL_EPISODES  = 50
N_PACKETS      = 100
LOG_INTERVAL   = 50
ADAPT_EPISODES = 50
ADAPT_DR       = 0.6

TOPOLOGIES = [
    ("abilene",   "data/ABI/Topology.txt",   "data/ABI/TrafficMatrix",   "Abilene"),
    ("geant",     "data/GEA/Topology.txt",   "data/GEA/TrafficMatrix",   "GEANT"),
    ("brain",     "data/BRA/Topology.txt",   "data/BRA/TrafficMatrix",   "BRAIN"),
    ("germany50", "data/GER50/Topology.txt", "data/GER50/TrafficMatrix", "Germany50"),
]

# (key, label, gnn_cls_or_None, init_type, is_gnn)
#   gnn_cls=None + is_gnn=True  → ScaIR baseline (SubGNN default)
#   gnn_cls=None + is_gnn=False → fixed encoding
VARIANTS = [
    ("scair",          "ScaIR (one-hot, concat)",   None,                 "onehot",       True),
    ("deg_mean",       "Degree + Mean GNN",         PaperSubGNN,          "degree",       True),
    ("deg_dot",        "Degree + Dot-Attn GNN",     DotAttnSubGNN,        "degree",       True),
    ("deg_lattn",      "Degree + Learn-Attn GNN",   LearnableAttnSubGNN,  "degree",       True),
    ("bet_mean",       "Betw. + Mean GNN",           PaperSubGNN,          "betweenness",  True),
    ("bet_dot",        "Betw. + Dot-Attn GNN",       DotAttnSubGNN,        "betweenness",  True),
    ("bet_lattn",      "Betw. + Learn-Attn GNN",     LearnableAttnSubGNN,  "betweenness",  True),
    ("sp_mean",        "ShortPath + Mean GNN",       PaperSubGNN,          "shortestpath", True),
    ("sp_dot",         "ShortPath + Dot-Attn GNN",   DotAttnSubGNN,        "shortestpath", True),
    ("sp_lattn",       "ShortPath + Learn-Attn GNN", LearnableAttnSubGNN,  "shortestpath", True),
    ("deg_fixed",      "Degree (no GNN)",            None,                 "degree",       False),
    ("bet_fixed",      "Betw. (no GNN)",             None,                 "betweenness",  False),
    ("sp_fixed",       "ShortPath (no GNN)",         None,                 "shortestpath", False),
]

COLORS = {
    "scair":    "black",
    "deg_mean": "steelblue", "deg_dot": "dodgerblue",   "deg_lattn": "deepskyblue",
    "bet_mean": "firebrick",  "bet_dot": "tomato",       "bet_lattn": "salmon",
    "sp_mean":  "seagreen",   "sp_dot":  "mediumseagreen","sp_lattn": "lightgreen",
    "deg_fixed":"royalblue",  "bet_fixed":"crimson",      "sp_fixed": "forestgreen",
}
MARKERS = {
    "scair": "D",
    "deg_mean":"o","deg_dot":"s","deg_lattn":"^",
    "bet_mean":"o","bet_dot":"s","bet_lattn":"^",
    "sp_mean":"o", "sp_dot":"s","sp_lattn":"^",
    "deg_fixed":"p","bet_fixed":"p","sp_fixed":"p",
}
LINESTYLES = {
    "scair": "-",
    "deg_mean":"-","deg_dot":"-","deg_lattn":"-",
    "bet_mean":"--","bet_dot":"--","bet_lattn":"--",
    "sp_mean":"-.","sp_dot":"-.","sp_lattn":"-.",
    "deg_fixed":":","bet_fixed":":","sp_fixed":":",
}


# ---------------------------------------------------------------------------
# Topology copy helpers (for adaptation test)
# ---------------------------------------------------------------------------

def copy_topology(topo):
    from scair.environment import Topology
    adj = {k: list(v) for k, v in topo.adjacency.items()}
    return Topology(topo.num_nodes, adj, dict(topo.link_delays))


def add_node(topo, agents, cfg, new_id: int, connect_to: List[int]):
    from scair.agent import IRrAgent
    topo.num_nodes = new_id + 1
    topo.adjacency[new_id] = list(connect_to)
    for ex in connect_to:
        topo.adjacency[ex] = topo.adjacency[ex] + [new_id]
    new_agent = IRrAgent(new_id, list(connect_to), topo.num_nodes, cfg)
    agents.append(new_agent)
    for ex in connect_to:
        ag = agents[ex]
        if new_id not in ag.neighbours:
            ag.neighbours = ag.neighbours + [new_id]
            ag.degree = len(ag.neighbours)
            ag._nbr_to_idx = {n: i for i, n in enumerate(ag.neighbours)}
            ag.queue_lengths[new_id] = 0


def remove_link(topo, agents, cfg, u: int, v: int):
    from scair.agent import IRrAgent
    topo.adjacency[u] = [x for x in topo.adjacency[u] if x != v]
    topo.adjacency[v] = [x for x in topo.adjacency[v] if x != u]
    for node in (u, v):
        old = agents[node]
        new_ag = IRrAgent(node, topo.adjacency[node], topo.num_nodes, cfg)
        new_ag.sub_gnn = old.sub_gnn
        new_ag._owns_gnn = old._owns_gnn
        new_ag.tick = old.tick
        new_ag.sigma = old.sigma
        agents[node] = new_ag


# ---------------------------------------------------------------------------
# OSPF
# ---------------------------------------------------------------------------

def _dijkstra(topo, source: int) -> Dict[int, int]:
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
    nh: Dict[int, int] = {}
    for dst in range(topo.num_nodes):
        if dst == source or dist[dst] == float("inf"):
            continue
        node = dst
        while prev.get(node, source) != source:
            node = prev[node]
        nh[dst] = node
    return nh


def eval_ospf(topo, cfg, episodes: List) -> float:
    routes = {n: _dijkstra(topo, n) for n in range(topo.num_nodes)}
    env = RoutingEnvironment(topo, cfg)
    times = []
    for packets in episodes:
        heap: list = []
        for pkt in packets:
            heapq.heappush(heap, (pkt.birth_time, pkt.pid, pkt, pkt.source, None))
        delivered: List[float] = []
        hop_counts: Dict[int, int] = defaultdict(int)
        queue_len: Dict[Tuple[int, int], int] = defaultdict(int)
        while heap:
            t, _, pkt, node, prev_node = heapq.heappop(heap)
            if prev_node is not None:
                queue_len[(prev_node, node)] = max(0, queue_len[(prev_node, node)] - 1)
            if node == pkt.destination:
                delivered.append(t - pkt.birth_time)
                continue
            if hop_counts[pkt.pid] >= cfg.max_hops:
                continue
            nxt = routes.get(node, {}).get(pkt.destination)
            if nxt is None:
                continue
            q = queue_len[(node, nxt)]
            cost = q * cfg.queue_time_per_packet + cfg.transmission_time
            queue_len[(node, nxt)] += 1
            hop_counts[pkt.pid] += 1
            heapq.heappush(heap, (t + cost, pkt.pid, pkt, nxt, node))
        times.append(float(np.mean(delivered)) if delivered else float("inf"))
    return float(np.mean(times))


# ---------------------------------------------------------------------------
# Episode pre-generation
# ---------------------------------------------------------------------------

def pregenerate_episodes(env, tms, n_train, n_eval, n_packets, seed):
    random.seed(seed)
    np.random.seed(seed)
    eps = []
    for ep in range(n_train + n_eval):
        tm = tms[ep % len(tms)]
        eps.append(env.generate_packets(tm, n_packets))
    return eps


# ---------------------------------------------------------------------------
# Build agents for a variant
# ---------------------------------------------------------------------------

def build_variant_agents(key, gnn_cls, init_type, is_gnn, topo, cfg, init_vs_cache):
    if is_gnn:
        if gnn_cls is None:
            # ScaIR baseline
            return build_agents(topo, cfg)
        ivs = init_vs_cache[init_type]
        return build_agents_topo_init(topo, cfg, gnn_cls, ivs)
    else:
        ivs = init_vs_cache[init_type]
        return build_agents_fixed_topo(topo, cfg, ivs)


# ---------------------------------------------------------------------------
# Train + eval one variant
# ---------------------------------------------------------------------------

def run_variant(label, agents, cfg, env, train_eps, eval_eps):
    torch.manual_seed(SEED)
    t0 = time.time()

    for ep, packets in enumerate(train_eps, start=1):
        stats = env.run_episode(packets, agents, training=True)
        if ep == 10:
            for ag in agents: ag.set_learning_rate(cfg.learning_rate)
        if ep % cfg.target_update_freq == 0:
            for ag in agents: ag.update_target()
        if ep % cfg.sigma_decay_freq == 0:
            for ag in agents: ag.decay_sigma()
        if ep % LOG_INTERVAL == 0 or ep == 1:
            print(f"    [{label}] ep {ep}/{TRAIN_EPISODES}  "
                  f"t={stats['avg_delivery_time']:.2f}ms  "
                  f"σ={agents[0].sigma:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    times = []
    for packets in eval_eps:
        stats = env.run_episode(packets, agents, training=False)
        times.append(stats["avg_delivery_time"])
    return float(np.mean(times)), float(np.std(times))


# ---------------------------------------------------------------------------
# Adaptation eval (no training)
# ---------------------------------------------------------------------------

def eval_agents(agents, cfg, env, episodes):
    times = []
    for packets in episodes:
        stats = env.run_episode(packets, agents, training=False)
        times.append(stats["avg_delivery_time"])
    return float(np.mean(times))


def adapt_and_eval(agents, cfg, env, adapt_eps, eval_eps):
    for packets in adapt_eps:
        env.run_episode(packets, agents, training=True)
        for ag in agents:
            if ag.tick % cfg.target_update_freq == 0:
                ag.update_target()
            if ag.tick % cfg.sigma_decay_freq == 0:
                ag.decay_sigma()
    return eval_agents(agents, cfg, env, eval_eps)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_dr_sweep(results, topo_label, out_path):
    drs = DR_VALUES
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"Topology-Init Experiment — {topo_label}", fontsize=13)

    ax = axes[0]
    ospf_vals = [results[dr]["ospf"] for dr in drs]
    ax.plot(drs, ospf_vals, "o--", color="green", label="OSPF", linewidth=2)
    for key, label, *_ in VARIANTS:
        times = [results[dr][key]["avg"] for dr in drs]
        ax.plot(drs, times, LINESTYLES[key], color=COLORS[key],
                marker=MARKERS[key], label=label, linewidth=1.5, markersize=5)
    ax.set_xlabel("D_r"); ax.set_ylabel("Avg Delivery Time (ms)")
    ax.set_title("Delivery Time"); ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3); ax.set_xticks(drs)

    ax = axes[1]
    for key, label, *_ in VARIANTS:
        gains = [(o - results[dr][key]["avg"]) / max(o, 1e-6) * 100
                 for o, dr in zip(ospf_vals, drs)]
        ax.plot(drs, gains, LINESTYLES[key], color=COLORS[key],
                marker=MARKERS[key], label=label, linewidth=1.5, markersize=5)
    ax.axhline(0, color="green", linestyle="--", linewidth=1.5, label="OSPF")
    ax.set_xlabel("D_r"); ax.set_ylabel("Improvement over OSPF (%)")
    ax.set_title("Gain vs OSPF"); ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3); ax.set_xticks(drs)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


def plot_adaptation(adapt_results, out_path):
    scenarios = list(adapt_results.keys())
    variant_keys = [k for k, *_ in VARIANTS]
    n_sc = len(scenarios)

    fig, axes = plt.subplots(1, n_sc, figsize=(8 * n_sc, 5), sharey=False)
    fig.suptitle(f"Topology Adaptation — Abilene D_r={ADAPT_DR}", fontsize=13)
    if n_sc == 1:
        axes = [axes]

    for ax, sc in zip(axes, scenarios):
        sc_data = adapt_results[sc]
        x = [0, ADAPT_EPISODES]
        for key in variant_keys:
            label = next(lbl for k, lbl, *_ in VARIANTS if k == key)
            y = [sc_data[key]["zero_shot"], sc_data[key]["after_adapt"]]
            ax.plot(x, y, LINESTYLES[key], color=COLORS[key],
                    marker=MARKERS[key], label=label, linewidth=1.5, markersize=6)
        ax.axhline(sc_data["ospf_after"], color="green", linestyle="--",
                   linewidth=1.5, label="OSPF (new topo)")
        ax.axhline(sc_data["ospf_before"], color="grey", linestyle=":",
                   linewidth=1.2, label="OSPF (original)")
        ax.set_xlabel("Adaptation episodes"); ax.set_ylabel("Avg Delivery Time (ms)")
        ax.set_title(sc); ax.legend(fontsize=6); ax.grid(alpha=0.3)
        ax.set_xticks(x)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(RESULTS, exist_ok=True)

    all_topo_results = {}

    # ---- track Abilene DR=ADAPT_DR agents for adaptation test ----
    abilene_adapt_agents: Dict[str, list] = {}
    abilene_adapt_topo = None
    abilene_adapt_cfg  = None

    for topo_key, topo_file, tm_dir, topo_label in TOPOLOGIES:
        print(f"\n{'#'*70}")
        print(f"  {topo_label}  ({topo_file})")
        print(f"{'#'*70}", flush=True)

        topo    = load_topology(topo_file)
        raw_tms = load_all_traffic_matrices(tm_dir, topo.num_nodes)
        tms     = [normalise_tm(tm) for tm in raw_tms]
        print(f"  {topo.num_nodes} nodes, {len(tms)} TMs", flush=True)

        base_cfg = ScaIRConfig()
        max_deg  = max(len(v) for v in topo.adjacency.values())
        if topo.num_nodes > base_cfg.max_nodes:  base_cfg.max_nodes  = topo.num_nodes
        if max_deg        > base_cfg.max_degree: base_cfg.max_degree = max_deg
        base_cfg.action_method = "ucb"

        # Pre-compute topology init vectors once per topology
        print("  Computing topology features...", flush=True)
        init_vs_cache = {
            "degree":      compute_init_vectors(topo, "degree",      base_cfg.feature_length),
            "betweenness": compute_init_vectors(topo, "betweenness", base_cfg.feature_length),
            "shortestpath":compute_init_vectors(topo, "shortestpath",base_cfg.feature_length),
        }

        topo_results = {}

        all_dr_values = list(DR_VALUES)
        if topo_key == "abilene" and ADAPT_DR not in all_dr_values:
            all_dr_values = all_dr_values + [ADAPT_DR]

        for dr in all_dr_values:
            print(f"\n{'='*70}")
            print(f"  [{topo_label}] D_r = {dr}", flush=True)
            print(f"{'='*70}", flush=True)

            cfg = copy.copy(base_cfg)
            cfg.distribution_ratio  = dr
            cfg.packets_per_episode = N_PACKETS

            env_gen = RoutingEnvironment(topo, cfg)
            all_eps = pregenerate_episodes(env_gen, tms, TRAIN_EPISODES,
                                           EVAL_EPISODES, N_PACKETS, SEED)
            train_eps = all_eps[:TRAIN_EPISODES]
            eval_eps  = all_eps[TRAIN_EPISODES:]

            ospf_val = eval_ospf(topo, cfg, eval_eps)
            print(f"  OSPF: {ospf_val:.3f} ms", flush=True)

            dr_results = {"ospf": ospf_val}

            for key, label, gnn_cls, init_type, is_gnn in VARIANTS:
                torch.manual_seed(SEED)
                agents = build_variant_agents(key, gnn_cls, init_type, is_gnn,
                                              topo, cfg, init_vs_cache)
                env = RoutingEnvironment(topo, cfg)
                avg, std = run_variant(label, agents, cfg, env, train_eps, eval_eps)
                gain = (ospf_val - avg) / max(ospf_val, 1e-6) * 100
                print(f"    {label}: {avg:.3f} ms ({gain:+.1f}% vs OSPF)", flush=True)
                dr_results[key] = {"avg": avg, "std": std}

                # Save Abilene ADAPT_DR agents for adaptation test
                if topo_key == "abilene" and abs(dr - ADAPT_DR) < 1e-9:
                    abilene_adapt_agents[key] = agents

            if dr in DR_VALUES:
                topo_results[dr] = dr_results

        all_topo_results[topo_key] = topo_results

        # Save topology results JSON
        json_path = os.path.join(RESULTS, f"{topo_key}_results.json")
        with open(json_path, "w") as f:
            json.dump({"topology": topo_label, "config": {
                "dr_values": DR_VALUES, "train_episodes": TRAIN_EPISODES,
                "eval_episodes": EVAL_EPISODES, "n_packets": N_PACKETS,
                "seed": SEED, "action_method": "ucb"},
                "results": {str(dr): r for dr, r in topo_results.items()}}, f, indent=2)
        print(f"  Saved: {json_path}", flush=True)

        plot_dr_sweep(topo_results, topo_label,
                      os.path.join(RESULTS, f"{topo_key}_dr_sweep.png"))

        if topo_key == "abilene":
            abilene_adapt_topo = topo
            abilene_adapt_cfg  = copy.copy(base_cfg)
            abilene_adapt_cfg.distribution_ratio  = ADAPT_DR
            abilene_adapt_cfg.packets_per_episode = N_PACKETS
            abilene_adapt_init_vs = init_vs_cache

    # -----------------------------------------------------------------------
    # Topology adaptation test (Abilene only)
    # -----------------------------------------------------------------------
    print(f"\n{'#'*70}")
    print(f"  Topology Adaptation Test — Abilene D_r={ADAPT_DR}")
    print(f"{'#'*70}", flush=True)

    topo = abilene_adapt_topo
    cfg  = abilene_adapt_cfg

    raw_tms = load_all_traffic_matrices("data/ABI/TrafficMatrix", topo.num_nodes)
    tms     = [normalise_tm(tm) for tm in raw_tms]

    # Generate episodes for adaptation phase
    random.seed(SEED + 999); np.random.seed(SEED + 999)
    adapt_tms_base = tms[TRAIN_EPISODES:]
    env_base = RoutingEnvironment(topo, cfg)
    zero_shot_eps = []
    adapt_train_eps = []
    adapt_eval_eps  = []
    for i in range(EVAL_EPISODES):
        zero_shot_eps.append(env_base.generate_packets(
            adapt_tms_base[i % len(adapt_tms_base)], N_PACKETS))
    for i in range(ADAPT_EPISODES):
        adapt_train_eps.append(env_base.generate_packets(
            adapt_tms_base[(EVAL_EPISODES + i) % len(adapt_tms_base)], N_PACKETS))
    for i in range(EVAL_EPISODES):
        adapt_eval_eps.append(env_base.generate_packets(
            adapt_tms_base[(EVAL_EPISODES + ADAPT_EPISODES + i) % len(adapt_tms_base)],
            N_PACKETS))

    SCENARIOS = [
        ("Add node 11 (→0,→8)", add_node,    {"new_id": 11, "connect_to": [0, 8]}),
        ("Remove link 3-6",     remove_link,  {"u": 3, "v": 6}),
    ]

    adapt_results = {}

    for sc_name, sc_fn, sc_kwargs in SCENARIOS:
        print(f"\n  Scenario: {sc_name}", flush=True)
        adapt_results[sc_name] = {}

        # OSPF on original topology
        ospf_before = eval_ospf(topo, cfg, zero_shot_eps)

        sc_agents_results = {}
        for key, label, gnn_cls, init_type, is_gnn in VARIANTS:
            # Clone topology for this scenario
            topo_mod = copy_topology(topo)
            cfg_mod  = copy.copy(cfg)
            if topo_mod.num_nodes > cfg_mod.max_nodes:
                cfg_mod.max_nodes = topo_mod.num_nodes

            # Deep-copy agents from trained Abilene agents
            import copy as copy_mod
            agents_mod = copy_mod.deepcopy(abilene_adapt_agents[key])

            # Apply topology change
            sc_fn(topo_mod, agents_mod, cfg_mod, **sc_kwargs)

            env_mod = RoutingEnvironment(topo_mod, cfg_mod)

            # Zero-shot eval
            zs = eval_agents(agents_mod, cfg_mod, env_mod, zero_shot_eps)

            # Adapt then eval
            after = adapt_and_eval(agents_mod, cfg_mod, env_mod,
                                   adapt_train_eps, adapt_eval_eps)

            gain_zs    = (ospf_before - zs)    / max(ospf_before, 1e-6) * 100
            gain_after = (ospf_before - after)  / max(ospf_before, 1e-6) * 100
            print(f"    {label}: zero-shot={zs:.3f} ({gain_zs:+.1f}%)  "
                  f"after {ADAPT_EPISODES} eps={after:.3f} ({gain_after:+.1f}%)", flush=True)
            sc_agents_results[key] = {"zero_shot": zs, "after_adapt": after}

        # OSPF on modified topology
        topo_for_ospf = copy_topology(topo)
        dummy_agents = []
        sc_fn(topo_for_ospf, dummy_agents, cfg, **sc_kwargs)
        ospf_after = eval_ospf(topo_for_ospf, cfg, adapt_eval_eps)

        adapt_results[sc_name] = {
            **sc_agents_results,
            "ospf_before": ospf_before,
            "ospf_after":  ospf_after,
        }

    adapt_path = os.path.join(RESULTS, "adaptation_results.json")
    with open(adapt_path, "w") as f:
        json.dump(adapt_results, f, indent=2, default=float)
    print(f"\nSaved: {adapt_path}", flush=True)

    plot_adaptation(adapt_results,
                    os.path.join(RESULTS, "adaptation_plot.png"))

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("  SUMMARY — Avg Delivery Time (ms) across all topologies and D_r")
    print(f"{'='*80}")
    for topo_key, _, _, topo_label in TOPOLOGIES:
        print(f"\n  {topo_label}:")
        res = all_topo_results[topo_key]
        header = f"  {'Variant':<30}" + "".join(f"  D_r={dr}" for dr in DR_VALUES)
        print(header)
        for key, label, *_ in [("OSPF", "OSPF")] + [(k, lbl) for k, lbl, *_ in VARIANTS]:
            row = f"  {label:<30}"
            for dr in DR_VALUES:
                if key == "OSPF":
                    row += f"  {res[dr]['ospf']:>8.3f}"
                else:
                    row += f"  {res[dr][key]['avg']:>8.3f}"
            print(row)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
