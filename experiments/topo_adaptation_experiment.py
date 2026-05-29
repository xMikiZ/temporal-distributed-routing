#!/usr/bin/env python3
"""
Topology Adaptation Test (Abilene, D_r=0.6) — standalone follow-up to Exp 9.

Trains all 13 Exp-9 variants on Abilene with D_r=0.6 (200 episodes),
then tests zero-shot performance and short-term adaptation (50 episodes)
under two topology changes:
  A) Add node 11 connected to nodes 0 and 8
  B) Remove link 3-6

Outputs
-------
  results/09_topo_init/adaptation_results.json
  results/09_topo_init/adaptation_plot.png
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
from scair.environment import RoutingEnvironment, Topology
from scair.models import (PaperSubGNN, DotAttnSubGNN, LearnableAttnSubGNN,
                           NeighborMaskSubGNN)
from scair.topology_features import compute_init_vectors
from train import (build_agents, build_agents_topo_init,
                   build_agents_fixed_topo)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS        = "results/09_topo_init"
SEED           = 42
ADAPT_DR       = 0.6
TRAIN_EPISODES = 200
EVAL_EPISODES  = 50
ADAPT_EPISODES = 50
N_PACKETS      = 100
LOG_INTERVAL   = 50

TOPO_FILE = "data/ABI/Topology.txt"
TM_DIR    = "data/ABI/TrafficMatrix"

VARIANTS = [
    ("scair",     "ScaIR (one-hot, concat)",    None,                "onehot",       True),
    ("deg_mean",  "Degree + Mean GNN",           PaperSubGNN,         "degree",       True),
    ("deg_dot",   "Degree + Dot-Attn GNN",       DotAttnSubGNN,       "degree",       True),
    ("deg_lattn", "Degree + Learn-Attn GNN",     LearnableAttnSubGNN, "degree",       True),
    ("bet_mean",  "Betw. + Mean GNN",             PaperSubGNN,         "betweenness",  True),
    ("bet_dot",   "Betw. + Dot-Attn GNN",         DotAttnSubGNN,       "betweenness",  True),
    ("bet_lattn", "Betw. + Learn-Attn GNN",       LearnableAttnSubGNN, "betweenness",  True),
    ("sp_mean",   "ShortPath + Mean GNN",         PaperSubGNN,         "shortestpath", True),
    ("sp_dot",    "ShortPath + Dot-Attn GNN",     DotAttnSubGNN,       "shortestpath", True),
    ("sp_lattn",  "ShortPath + Learn-Attn GNN",   LearnableAttnSubGNN, "shortestpath", True),
    ("deg_fixed", "Degree (no GNN)",              None,                "degree",       False),
    ("bet_fixed", "Betw. (no GNN)",               None,                "betweenness",  False),
    ("sp_fixed",  "ShortPath (no GNN)",           None,                "shortestpath", False),
]

COLORS = {
    "scair":    "black",
    "deg_mean": "steelblue",  "deg_dot": "dodgerblue",    "deg_lattn": "deepskyblue",
    "bet_mean": "firebrick",  "bet_dot": "tomato",         "bet_lattn": "salmon",
    "sp_mean":  "seagreen",   "sp_dot":  "mediumseagreen", "sp_lattn":  "lightgreen",
    "deg_fixed":"royalblue",  "bet_fixed":"crimson",       "sp_fixed":  "forestgreen",
}
MARKERS    = {"scair":"D", **{k:"o" for k,*_ in VARIANTS[1:4]},
              **{k:"o" for k,*_ in VARIANTS[4:7]},
              **{k:"o" for k,*_ in VARIANTS[7:10]},
              **{k:"p" for k,*_ in VARIANTS[10:]}}
LINESTYLES = {
    "scair":"-",
    "deg_mean":"-","deg_dot":"-","deg_lattn":"-",
    "bet_mean":"--","bet_dot":"--","bet_lattn":"--",
    "sp_mean":"-.","sp_dot":"-.","sp_lattn":"-.",
    "deg_fixed":":","bet_fixed":":","sp_fixed":":",
}


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

def copy_topology(topo):
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
# OSPF baseline
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


def eval_ospf(topo, cfg, episodes):
    routes = {n: _dijkstra(topo, n) for n in range(topo.num_nodes)}
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
# Helpers
# ---------------------------------------------------------------------------

def pregenerate_episodes(env, tms, n, n_packets, seed_offset=0):
    random.seed(SEED + seed_offset)
    np.random.seed(SEED + seed_offset)
    return [env.generate_packets(tms[i % len(tms)], n_packets) for i in range(n)]


def build_variant_agents(key, gnn_cls, init_type, is_gnn, topo, cfg, init_vs):
    if is_gnn:
        if gnn_cls is None:
            return build_agents(topo, cfg)
        return build_agents_topo_init(topo, cfg, gnn_cls, init_vs[init_type])
    else:
        return build_agents_fixed_topo(topo, cfg, init_vs[init_type])


def train_variant(label, agents, cfg, env, train_eps):
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
                  f"sigma={agents[0].sigma:.2f}  ({time.time()-t0:.0f}s)", flush=True)


def eval_agents(agents, env, episodes):
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
    return eval_agents(agents, env, eval_eps)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_adaptation(adapt_results, out_path):
    scenarios = list(adapt_results.keys())
    n_sc = len(scenarios)
    fig, axes = plt.subplots(1, n_sc, figsize=(8 * n_sc, 5), sharey=False)
    fig.suptitle(f"Topology Adaptation — Abilene D_r={ADAPT_DR}", fontsize=13)
    if n_sc == 1:
        axes = [axes]

    for ax, sc in zip(axes, scenarios):
        sc_data = adapt_results[sc]
        x = [0, ADAPT_EPISODES]
        for key, label, *_ in VARIANTS:
            y = [sc_data[key]["zero_shot"], sc_data[key]["after_adapt"]]
            ax.plot(x, y, LINESTYLES[key], color=COLORS[key],
                    marker=MARKERS.get(key, "o"), label=label, linewidth=1.5, markersize=6)
        ax.axhline(sc_data["ospf_after"], color="green", linestyle="--",
                   linewidth=1.5, label="OSPF (new topo)")
        ax.axhline(sc_data["ospf_before"], color="grey", linestyle=":",
                   linewidth=1.2, label="OSPF (original)")
        ax.set_xlabel("Adaptation episodes")
        ax.set_ylabel("Avg Delivery Time (ms)")
        ax.set_title(sc)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
        ax.set_xticks(x)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(RESULTS, exist_ok=True)

    print(f"Loading Abilene topology...")
    topo = load_topology(TOPO_FILE)
    raw_tms = load_all_traffic_matrices(TM_DIR, topo.num_nodes)
    tms = [normalise_tm(tm) for tm in raw_tms]
    print(f"  {topo.num_nodes} nodes, {len(tms)} TMs", flush=True)

    cfg = ScaIRConfig()
    max_deg = max(len(v) for v in topo.adjacency.values())
    if topo.num_nodes > cfg.max_nodes:  cfg.max_nodes  = topo.num_nodes
    if max_deg        > cfg.max_degree: cfg.max_degree = max_deg
    cfg.action_method       = "ucb"
    cfg.distribution_ratio  = ADAPT_DR
    cfg.packets_per_episode = N_PACKETS

    print("  Computing topology features...", flush=True)
    init_vs = {
        "degree":       compute_init_vectors(topo, "degree",       cfg.feature_length),
        "betweenness":  compute_init_vectors(topo, "betweenness",  cfg.feature_length),
        "shortestpath": compute_init_vectors(topo, "shortestpath", cfg.feature_length),
    }

    # Pre-generate training and evaluation episodes
    env_gen = RoutingEnvironment(topo, cfg)
    train_eps = pregenerate_episodes(env_gen, tms, TRAIN_EPISODES, N_PACKETS, seed_offset=0)
    eval_eps  = pregenerate_episodes(env_gen, tms, EVAL_EPISODES,  N_PACKETS, seed_offset=1000)

    # Adaptation episodes (separate seed to avoid overlap)
    zero_shot_eps  = pregenerate_episodes(env_gen, tms, EVAL_EPISODES,  N_PACKETS, seed_offset=2000)
    adapt_train_eps = pregenerate_episodes(env_gen, tms, ADAPT_EPISODES, N_PACKETS, seed_offset=3000)
    adapt_eval_eps  = pregenerate_episodes(env_gen, tms, EVAL_EPISODES,  N_PACKETS, seed_offset=4000)

    print(f"\nOSPF baseline (training topology): ", end="", flush=True)
    ospf_base = eval_ospf(topo, cfg, eval_eps)
    print(f"{ospf_base:.3f} ms", flush=True)

    # ---- Train all 13 variants ----
    print(f"\n{'='*70}")
    print(f"  Training 13 variants — Abilene D_r={ADAPT_DR}")
    print(f"{'='*70}", flush=True)

    trained_agents: Dict[str, list] = {}
    for key, label, gnn_cls, init_type, is_gnn in VARIANTS:
        torch.manual_seed(SEED)
        agents = build_variant_agents(key, gnn_cls, init_type, is_gnn, topo, cfg, init_vs)
        env = RoutingEnvironment(topo, cfg)
        train_variant(label, agents, cfg, env, train_eps)
        avg = eval_agents(agents, env, eval_eps)
        gain = (ospf_base - avg) / max(ospf_base, 1e-6) * 100
        print(f"  {label}: {avg:.3f} ms ({gain:+.1f}% vs OSPF)", flush=True)
        trained_agents[key] = agents

    # ---- Adaptation scenarios ----
    SCENARIOS = [
        ("Add node 11 (->0, ->8)", add_node,    {"new_id": 11, "connect_to": [0, 8]}),
        ("Remove link 3-6",        remove_link,  {"u": 3, "v": 6}),
    ]

    adapt_results = {}

    for sc_name, sc_fn, sc_kwargs in SCENARIOS:
        print(f"\n{'='*70}")
        print(f"  Scenario: {sc_name}", flush=True)
        print(f"{'='*70}", flush=True)

        # OSPF on original topology
        ospf_before = eval_ospf(topo, cfg, zero_shot_eps)

        sc_agents_results = {}
        for key, label, gnn_cls, init_type, is_gnn in VARIANTS:
            topo_mod = copy_topology(topo)
            cfg_mod  = copy.copy(cfg)
            if topo_mod.num_nodes > cfg_mod.max_nodes:
                cfg_mod.max_nodes = topo_mod.num_nodes

            import copy as copy_mod
            agents_mod = copy_mod.deepcopy(trained_agents[key])

            sc_fn(topo_mod, agents_mod, cfg_mod, **sc_kwargs)

            env_mod = RoutingEnvironment(topo_mod, cfg_mod)

            zs    = eval_agents(agents_mod, env_mod, zero_shot_eps)
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

    # ---- Save & plot ----
    out_path = os.path.join(RESULTS, "adaptation_results.json")
    with open(out_path, "w") as f:
        json.dump(adapt_results, f, indent=2, default=float)
    print(f"\nSaved: {out_path}", flush=True)

    plot_adaptation(adapt_results, os.path.join(RESULTS, "adaptation_plot.png"))

    # ---- Summary table ----
    print(f"\n{'='*70}")
    print(f"  ADAPTATION SUMMARY — Abilene D_r={ADAPT_DR}")
    print(f"{'='*70}")
    for sc_name in adapt_results:
        sc = adapt_results[sc_name]
        print(f"\n  {sc_name}  (OSPF before={sc['ospf_before']:.3f}  after={sc['ospf_after']:.3f})")
        print(f"  {'Variant':<35}  {'Zero-shot':>10}  {'After-adapt':>12}")
        for key, label, *_ in VARIANTS:
            print(f"  {label:<35}  {sc[key]['zero_shot']:>10.3f}  {sc[key]['after_adapt']:>12.3f}")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
