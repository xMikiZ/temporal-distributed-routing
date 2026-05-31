#!/usr/bin/env python3
"""
Replication of ScaIR §5.2.6 online adaptability experiment on Abilene.

The paper trains on hot-spot 0→8 (3×3 net) then at episode 200 switches to
hot-spot 2→6 (opposite diagonal). We replicate this on Abilene with two pairs
that have zero shared directed edges — opposite sides of the network:

  Pair A: 7 → 8  (path: 7–1–4–5–2–8,   5 hops, west corridor)
  Pair B: 9 → 10 (path: 9–2–5–3–0–10,  5 hops, east corridor)

Protocol (matching the paper):
  1. Pre-train ScaIR for 300 episodes on hot-spot A  (σ decays 0.9→0.1)
  2. Online phase A (150 eps): continue on hot-spot A with σ fixed at 0.1
  3. Online phase B (150 eps): switch to hot-spot B,  σ still 0.1

OSPF is evaluated on the same packets as ScaIR each episode for comparison.
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
from scair.environment import RoutingEnvironment, Packet
from train import build_agents
from experiments.optimal_comparison import run_ospf_episode

RESULTS      = "results/online_adaptability"
SEED         = 42
TOPO_FILE    = "data/ABI/Topology.txt"
TM_DIR       = "data/ABI/TrafficMatrix"
DR           = 0.9
N_PACKETS    = 50
PRETRAIN_EPS = 300
PHASE_A_EPS  = 150
PHASE_B_EPS  = 150
PAIR_A       = (7, 8)
PAIR_B       = (9, 10)


def generate_packets(env, tm, n_packets, hotspot_pair, gen_interval, dr):
    n = env.topo.num_nodes
    tm_f = tm.astype(float).copy()
    np.fill_diagonal(tm_f, 0.0)
    total = tm_f.sum()
    probs = (tm_f / total).flatten() if total > 0 else None
    if probs is None:
        probs = np.ones(n * n) / max(n * n - n, 1)
        for i in range(n): probs[i * n + i] = 0.0
    src_hs, dst_hs = hotspot_pair
    packets = []
    t = 0.0
    for pid in range(n_packets):
        t += np.random.exponential(gen_interval)
        if random.random() < dr:
            src, dst = src_hs, dst_hs
        else:
            idx = int(np.random.choice(n * n, p=probs))
            src, dst = divmod(idx, n)
        packets.append(Packet(pid, src, dst, t))
    return packets


def main():
    os.makedirs(RESULTS, exist_ok=True)

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

    print("Online Adaptability — Abilene")
    print(f"Pair A: {PAIR_A[0]}→{PAIR_A[1]}   Pair B: {PAIR_B[0]}→{PAIR_B[1]}   D_r={DR}\n")

    # ----------------------------------------------------------------
    # Phase 0: pre-train on hot-spot A
    # ----------------------------------------------------------------
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    agents = build_agents(topo, cfg)
    env    = RoutingEnvironment(topo, cfg)

    print(f"--- Pre-training: {PRETRAIN_EPS} eps on {PAIR_A[0]}→{PAIR_A[1]} ---", flush=True)
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
    print(f"\nPre-training done. σ settled at {sigma_final:.2f}")

    # ----------------------------------------------------------------
    # Online test: phase A then phase B, σ frozen
    # ----------------------------------------------------------------
    print(f"\n--- Online test (σ={sigma_final:.2f} fixed) ---", flush=True)
    scair_curve, ospf_curve = [], []
    total_online = PHASE_A_EPS + PHASE_B_EPS
    t0 = time.time()

    for ep in range(1, total_online + 1):
        pair    = PAIR_A if ep <= PHASE_A_EPS else PAIR_B
        packets = generate_packets(env, tms[(ep-1) % len(tms)], N_PACKETS,
                                   pair, cfg.generation_interval, DR)

        stats  = env.run_episode(packets, agents, training=True)
        ospf_t = run_ospf_episode(topo, cfg, packets)
        scair_curve.append(stats["avg_delivery_time"])
        ospf_curve.append(ospf_t)

        if ep % cfg.target_update_freq == 0:
            for ag in agents: ag.update_target()
        # NO sigma decay during online phase

        if ep % 20 == 0 or ep in (1, PHASE_A_EPS, PHASE_A_EPS + 1):
            phase = "A" if ep <= PHASE_A_EPS else "B"
            print(f"  online ep {ep:3d} [{phase}] {pair}  "
                  f"ScaIR={scair_curve[-1]:.2f}ms  OSPF={ospf_curve[-1]:.2f}ms  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # Summary
    def tail_mean(curve, start, end):
        window = curve[max(start, end-30):end]
        return float(np.mean(window))

    pA_s = tail_mean(scair_curve, 0, PHASE_A_EPS)
    pA_o = tail_mean(ospf_curve,  0, PHASE_A_EPS)
    pB_s = tail_mean(scair_curve, PHASE_A_EPS, total_online)
    pB_o = tail_mean(ospf_curve,  PHASE_A_EPS, total_online)

    print(f"\n{'─'*55}")
    print(f"  Phase A (last 30 ep):  ScaIR={pA_s:.3f}ms  OSPF={pA_o:.3f}ms  "
          f"ScaIR {'<' if pA_s<pA_o else '>'} OSPF")
    print(f"  Phase B (last 30 ep):  ScaIR={pB_s:.3f}ms  OSPF={pB_o:.3f}ms  "
          f"ScaIR {'<' if pB_s<pB_o else '>'} OSPF")
    print(f"  Adaptation: ScaIR goes from {scair_curve[PHASE_A_EPS-1]:.2f}ms → "
          f"{scair_curve[PHASE_A_EPS]:.2f}ms at switch, then → {pB_s:.2f}ms")
    print(f"{'─'*55}")

    # Save
    out = {"pair_A": PAIR_A, "pair_B": PAIR_B, "dr": DR,
           "pretrain_eps": PRETRAIN_EPS, "sigma_online": sigma_final,
           "scair_curve": scair_curve, "ospf_curve": ospf_curve,
           "phase_A": {"scair": pA_s, "ospf": pA_o},
           "phase_B": {"scair": pB_s, "ospf": pB_o}}
    with open(os.path.join(RESULTS, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    eps = np.arange(1, total_online + 1)
    win = 8

    def smooth(x):
        return np.convolve(x, np.ones(win)/win, mode='valid')

    ax.plot(eps, scair_curve, color="#e67e22", alpha=0.2, linewidth=0.7)
    ax.plot(eps[win-1:], smooth(scair_curve), color="#e67e22",
            linewidth=2, label="ScaIR (ε-greedy)")
    ax.plot(eps, ospf_curve, color="#3498db", alpha=0.2, linewidth=0.7)
    ax.plot(eps[win-1:], smooth(ospf_curve), color="#3498db",
            linewidth=2, label="OSPF")

    ax.axvline(PHASE_A_EPS, color="black", linestyle="--", linewidth=1.5)
    ylim = ax.get_ylim()
    ax.text(PHASE_A_EPS + 2, ylim[1] * 0.96,
            f"Hot-spot switches\n{PAIR_A[0]}→{PAIR_A[1]}  →  {PAIR_B[0]}→{PAIR_B[1]}",
            fontsize=9, va="top")
    ax.text(PHASE_A_EPS * 0.45, ylim[1] * 0.88,
            f"Phase A\nhot-spot {PAIR_A[0]}→{PAIR_A[1]}",
            ha="center", fontsize=10, color="#555")
    ax.text(PHASE_A_EPS + PHASE_B_EPS * 0.5, ylim[1] * 0.88,
            f"Phase B\nhot-spot {PAIR_B[0]}→{PAIR_B[1]}",
            ha="center", fontsize=10, color="#555")

    ax.set_xlabel("Online episode"); ax.set_ylabel("Avg Delivery Time (ms)")
    ax.set_title(f"Online Adaptability — Abilene  D_r={DR}  "
                 f"(pre-trained {PRETRAIN_EPS} eps, σ={sigma_final:.2f} fixed online)")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS, "online_adaptability.png")
    plt.savefig(plot_path, dpi=150); plt.close(fig)
    print(f"Saved: {plot_path}")
    print("Done.")


if __name__ == "__main__":
    main()
