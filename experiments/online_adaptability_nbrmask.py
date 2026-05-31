#!/usr/bin/env python3
"""
Adds the NbrMask (no-GNN) variant to the online adaptability comparison.

Runs the same protocol as online_adaptability.py with NeighborMaskSubGNN:
  - V_n = fixed binary mask (1 at each neighbour's index, 0 elsewhere)
  - No message passing, no learned topology encoding
  - Q-network trains normally; only GNN is replaced by a static mask

Merges results into the existing results.json and replots.
"""

import json
import os
import random
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scair.config import ScaIRConfig
from scair.data_loader import load_all_traffic_matrices, load_topology, normalise_tm
from scair.models import NeighborMaskSubGNN
from scair.environment import RoutingEnvironment
from train import build_agents_no_gnn
from experiments.online_adaptability import generate_packets, RESULTS, SEED, TOPO_FILE, TM_DIR
from experiments.online_adaptability import DR, N_PACKETS, PRETRAIN_EPS, PHASE_A_EPS, PHASE_B_EPS
from experiments.online_adaptability import PAIR_A, PAIR_B
from experiments.optimal_comparison import run_ospf_episode


def run_nbrmask():
    topo    = load_topology(TOPO_FILE)
    raw_tms = load_all_traffic_matrices(TM_DIR, topo.num_nodes)
    tms     = [normalise_tm(tm) for tm in raw_tms]

    cfg = ScaIRConfig()
    max_deg = max(len(v) for v in topo.adjacency.values())
    if topo.num_nodes > cfg.max_nodes:  cfg.max_nodes  = topo.num_nodes
    if max_deg        > cfg.max_degree: cfg.max_degree = max_deg
    cfg.action_method       = "epsilon_greedy"
    cfg.distribution_ratio  = 0.0
    cfg.packets_per_episode = N_PACKETS

    print("NbrMask variant — same protocol as ScaIR")
    print(f"Pre-train: {PRETRAIN_EPS} eps on {PAIR_A[0]}→{PAIR_A[1]}, then online switch\n")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    agents = build_agents_no_gnn(topo, cfg, NeighborMaskSubGNN)
    env    = RoutingEnvironment(topo, cfg)

    # Pre-train on hot-spot A
    print(f"--- Pre-training ---", flush=True)
    t0 = time.time()
    for ep in range(1, PRETRAIN_EPS + 1):
        packets = generate_packets(env, tms[(ep-1) % len(tms)], N_PACKETS,
                                   PAIR_A, cfg.generation_interval, DR)
        stats = env.run_episode(packets, agents, training=True)
        if ep == 10:
            for ag in agents: ag.set_learning_rate(cfg.learning_rate)
        if ep % cfg.target_update_freq == 0:
            for ag in agents: ag.update_target()
        if ep % cfg.sigma_decay_freq == 0:
            for ag in agents: ag.decay_sigma()
        if ep % 50 == 0 or ep == 1:
            print(f"  ep {ep:3d}/{PRETRAIN_EPS}  t={stats['avg_delivery_time']:.2f}ms  "
                  f"σ={agents[0].sigma:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    sigma_final = agents[0].sigma
    print(f"\nPre-training done. σ={sigma_final:.2f}")

    # Online test
    print(f"\n--- Online test (σ={sigma_final:.2f} fixed) ---", flush=True)
    nbrmask_curve = []
    total_online  = PHASE_A_EPS + PHASE_B_EPS
    t0 = time.time()

    for ep in range(1, total_online + 1):
        pair    = PAIR_A if ep <= PHASE_A_EPS else PAIR_B
        packets = generate_packets(env, tms[(ep-1) % len(tms)], N_PACKETS,
                                   pair, cfg.generation_interval, DR)
        stats = env.run_episode(packets, agents, training=True)
        nbrmask_curve.append(stats["avg_delivery_time"])

        if ep % cfg.target_update_freq == 0:
            for ag in agents: ag.update_target()

        if ep % 20 == 0 or ep in (1, PHASE_A_EPS, PHASE_A_EPS + 1):
            phase = "A" if ep <= PHASE_A_EPS else "B"
            print(f"  online ep {ep:3d} [{phase}] {pair}  "
                  f"NbrMask={nbrmask_curve[-1]:.2f}ms  ({time.time()-t0:.0f}s)", flush=True)

    def tail_mean(curve, start, end):
        return float(np.mean(curve[max(start, end-30):end]))

    pA = tail_mean(nbrmask_curve, 0, PHASE_A_EPS)
    pB = tail_mean(nbrmask_curve, PHASE_A_EPS, total_online)
    print(f"\nPhase A (last 30): {pA:.3f}ms   Phase B (last 30): {pB:.3f}ms")
    print(f"Switch: {nbrmask_curve[PHASE_A_EPS-1]:.2f}ms → {nbrmask_curve[PHASE_A_EPS]:.2f}ms → {pB:.2f}ms")

    return nbrmask_curve, pA, pB


def replot(data):
    scair   = data["scair_curve"]
    ospf    = data["ospf_curve"]
    nbrmask = data["nbrmask_curve"]
    total   = PHASE_A_EPS + PHASE_B_EPS
    eps     = np.arange(1, total + 1)
    win     = 8

    def smooth(x):
        return np.convolve(x, np.ones(win)/win, mode='valid')

    fig, ax = plt.subplots(figsize=(12, 5))

    for curve, color, label in [
        (scair,   "#e67e22", "ScaIR (with GNN)"),
        (nbrmask, "#9b59b6", "NbrMask (no GNN)"),
        (ospf,    "#3498db", "OSPF"),
    ]:
        ax.plot(eps, curve, color=color, alpha=0.18, linewidth=0.7)
        ax.plot(eps[win-1:], smooth(curve), color=color, linewidth=2, label=label)

    ax.axvline(PHASE_A_EPS, color="black", linestyle="--", linewidth=1.5)
    ylim = ax.get_ylim()
    ax.text(PHASE_A_EPS + 2, ylim[1] * 0.96,
            f"Hot-spot switches\n{PAIR_A[0]}→{PAIR_A[1]}  →  {PAIR_B[0]}→{PAIR_B[1]}",
            fontsize=9, va="top")
    ax.text(PHASE_A_EPS * 0.4, ylim[1] * 0.87,
            f"Phase A\n{PAIR_A[0]}→{PAIR_A[1]}",
            ha="center", fontsize=10, color="#555")
    ax.text(PHASE_A_EPS + PHASE_B_EPS * 0.5, ylim[1] * 0.87,
            f"Phase B\n{PAIR_B[0]}→{PAIR_B[1]}",
            ha="center", fontsize=10, color="#555")

    ax.set_xlabel("Online episode")
    ax.set_ylabel("Avg Delivery Time (ms)")
    ax.set_title(f"Online Adaptability — Abilene  D_r={DR}  "
                 f"(pre-trained {PRETRAIN_EPS} eps)")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS, "online_adaptability.png")
    plt.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def main():
    nbrmask_curve, pA, pB = run_nbrmask()

    # Load existing results and add NbrMask
    json_path = os.path.join(RESULTS, "results.json")
    with open(json_path) as f:
        data = json.load(f)

    data["nbrmask_curve"] = nbrmask_curve
    data["phase_A"]["nbrmask"] = pA
    data["phase_B"]["nbrmask"] = pB

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Updated: {json_path}")

    replot(data)
    print("Done.")


if __name__ == "__main__":
    main()
