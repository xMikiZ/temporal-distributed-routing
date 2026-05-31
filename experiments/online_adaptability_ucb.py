#!/usr/bin/env python3
"""
Adds a UCB variant to the online adaptability comparison.

UCB accumulates visit counts across all episodes and never resets them.
After 300 pre-training episodes, every action has been visited many times
and the exploration bonus is near zero — the policy is essentially pure
exploitation. When the hot-spot switches, UCB cannot actively explore
new routes; it only adapts through Q-value updates via experience replay.

Contrast with epsilon-greedy (sigma=0.1 fixed): always 10% random actions,
which may happen to sample routes better suited to the new hot-spot and
collect experience faster.

This script runs the UCB variant, merges into results.json, and replots
the three curves: ε-greedy ScaIR (existing), UCB ScaIR (new), OSPF.
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
from scair.environment import RoutingEnvironment
from train import build_agents
from experiments.online_adaptability import (
    generate_packets, RESULTS, SEED, TOPO_FILE, TM_DIR,
    DR, N_PACKETS, PRETRAIN_EPS, PHASE_A_EPS, PHASE_B_EPS,
    PAIR_A, PAIR_B,
)
from experiments.optimal_comparison import run_ospf_episode


def run_ucb():
    topo    = load_topology(TOPO_FILE)
    raw_tms = load_all_traffic_matrices(TM_DIR, topo.num_nodes)
    tms     = [normalise_tm(tm) for tm in raw_tms]

    cfg = ScaIRConfig()
    max_deg = max(len(v) for v in topo.adjacency.values())
    if topo.num_nodes > cfg.max_nodes:  cfg.max_nodes  = topo.num_nodes
    if max_deg        > cfg.max_degree: cfg.max_degree = max_deg
    cfg.action_method       = "ucb"       # <-- UCB, counts accumulate forever
    cfg.distribution_ratio  = 0.0
    cfg.packets_per_episode = N_PACKETS

    print("UCB variant — same protocol as ε-greedy run\n")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    agents = build_agents(topo, cfg)
    env    = RoutingEnvironment(topo, cfg)

    # Pre-train on hot-spot A
    print(f"--- Pre-training: {PRETRAIN_EPS} eps on {PAIR_A} ---", flush=True)
    t0 = time.time()
    for ep in range(1, PRETRAIN_EPS + 1):
        packets = generate_packets(env, tms[(ep-1) % len(tms)], N_PACKETS,
                                   PAIR_A, cfg.generation_interval, DR)
        stats = env.run_episode(packets, agents, training=True)
        if ep == 10:
            for ag in agents: ag.set_learning_rate(cfg.learning_rate)
        if ep % cfg.target_update_freq == 0:
            for ag in agents: ag.update_target()
        if ep % 50 == 0 or ep == 1:
            print(f"  ep {ep:3d}/{PRETRAIN_EPS}  t={stats['avg_delivery_time']:.2f}ms  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # Quick check: how large are UCB counts after pre-training?
    sample_counts = agents[0]._ucb_counts
    if sample_counts:
        dst = next(iter(sample_counts))
        total_visits = sum(sample_counts[dst])
        print(f"\nPre-training done. Example: node 0, dst={dst}, "
              f"total visits={total_visits} (bonus ≈ {2.0*(np.log(total_visits+1)/total_visits)**0.5:.4f})")

    # Online test — UCB counts are NOT reset; bonus is near-zero for known actions
    print(f"\n--- Online test (UCB, counts not reset) ---", flush=True)
    ucb_curve    = []
    total_online = PHASE_A_EPS + PHASE_B_EPS
    t0 = time.time()

    for ep in range(1, total_online + 1):
        pair    = PAIR_A if ep <= PHASE_A_EPS else PAIR_B
        packets = generate_packets(env, tms[(ep-1) % len(tms)], N_PACKETS,
                                   pair, cfg.generation_interval, DR)
        stats = env.run_episode(packets, agents, training=True)
        ucb_curve.append(stats["avg_delivery_time"])

        if ep % cfg.target_update_freq == 0:
            for ag in agents: ag.update_target()

        if ep % 20 == 0 or ep in (1, PHASE_A_EPS, PHASE_A_EPS + 1):
            phase = "A" if ep <= PHASE_A_EPS else "B"
            print(f"  online ep {ep:3d} [{phase}] {pair}  "
                  f"UCB={ucb_curve[-1]:.2f}ms  ({time.time()-t0:.0f}s)", flush=True)

    def tail_mean(curve, start, end):
        return float(np.mean(curve[max(start, end-30):end]))

    pA = tail_mean(ucb_curve, 0, PHASE_A_EPS)
    pB = tail_mean(ucb_curve, PHASE_A_EPS, total_online)
    print(f"\nPhase A (last 30): {pA:.3f}ms")
    print(f"Phase B (last 30): {pB:.3f}ms")
    print(f"Switch: {ucb_curve[PHASE_A_EPS-1]:.2f}ms → "
          f"{ucb_curve[PHASE_A_EPS]:.2f}ms → {pB:.2f}ms")

    return ucb_curve, pA, pB


def replot(data):
    eps_greedy = data["scair_curve"]
    ucb        = data["ucb_curve"]
    ospf       = data["ospf_curve"]
    nbrmask    = data.get("nbrmask_curve")
    total      = PHASE_A_EPS + PHASE_B_EPS
    eps        = np.arange(1, total + 1)
    win        = 8

    def smooth(x):
        return np.convolve(x, np.ones(win)/win, mode='valid')

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"Online Adaptability — Abilene  D_r={DR}  "
                 f"(pre-trained {PRETRAIN_EPS} eps)", fontsize=13)

    # Left: full curves
    ax = axes[0]
    curves = [
        (eps_greedy, "#e67e22", "ScaIR ε-greedy (σ=0.10 fixed)"),
        (ucb,        "#e74c3c", "ScaIR UCB (counts not reset)"),
        (ospf,       "#3498db", "OSPF"),
    ]
    if nbrmask:
        curves.insert(2, (nbrmask, "#9b59b6", "NbrMask ε-greedy"))

    for curve, color, label in curves:
        ax.plot(eps, curve, color=color, alpha=0.18, linewidth=0.7)
        ax.plot(eps[win-1:], smooth(curve), color=color, linewidth=2, label=label)

    ax.axvline(PHASE_A_EPS, color="black", linestyle="--", linewidth=1.5)
    ylim = ax.get_ylim()
    ax.text(PHASE_A_EPS + 2, ylim[1] * 0.96,
            f"Switch\n{PAIR_A[0]}→{PAIR_A[1]} to {PAIR_B[0]}→{PAIR_B[1]}",
            fontsize=9, va="top")
    ax.text(PHASE_A_EPS * 0.4, ylim[1] * 0.88,
            f"Phase A\n{PAIR_A[0]}→{PAIR_A[1]}", ha="center", fontsize=9, color="#555")
    ax.text(PHASE_A_EPS + PHASE_B_EPS * 0.5, ylim[1] * 0.88,
            f"Phase B\n{PAIR_B[0]}→{PAIR_B[1]}", ha="center", fontsize=9, color="#555")
    ax.set_xlabel("Online episode"); ax.set_ylabel("Avg Delivery Time (ms)")
    ax.set_title("Full online test")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Right: zoom on adaptation window (ep 130–200)
    ax = axes[1]
    zoom_start, zoom_end = PHASE_A_EPS - 20, PHASE_A_EPS + 60
    zoom_eps = eps[zoom_start:zoom_end]
    for curve, color, label in curves:
        ax.plot(zoom_eps, curve[zoom_start:zoom_end],
                color=color, linewidth=1.5, alpha=0.35)
        sm = smooth(curve)
        # smoothed within zoom window (careful with offsets)
        sm_start = max(0, zoom_start - win + 1)
        sm_eps_full = np.arange(win, total + 1)
        mask = (sm_eps_full >= zoom_start + 1) & (sm_eps_full <= zoom_end)
        ax.plot(sm_eps_full[mask], sm[mask - 1 + (zoom_start - sm_start)
                                      if False else mask],
                color=color, linewidth=2.5, label=label)

    # Simpler zoom plot without smoothing offset confusion
    ax.cla()
    for curve, color, label in curves:
        ax.plot(zoom_eps, curve[zoom_start:zoom_end],
                color=color, linewidth=2, label=label, alpha=0.85)
    ax.axvline(PHASE_A_EPS, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Online episode"); ax.set_ylabel("Avg Delivery Time (ms)")
    ax.set_title(f"Zoom: adaptation window (ep {zoom_start+1}–{zoom_end})")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS, "online_adaptability.png")
    plt.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


def main():
    ucb_curve, pA, pB = run_ucb()

    json_path = os.path.join(RESULTS, "results.json")
    with open(json_path) as f:
        data = json.load(f)

    data["ucb_curve"] = ucb_curve
    data["phase_A"]["ucb"] = pA
    data["phase_B"]["ucb"] = pB

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Updated: {json_path}")

    # Print comparison summary
    print(f"\n{'─'*60}")
    print(f"  {'Method':<28} {'Phase A':>10} {'Phase B':>10} {'Switch spike':>14}")
    print(f"{'─'*60}")
    for key, label in [("scair", "ε-greedy ScaIR"),
                        ("ucb",   "UCB ScaIR"),
                        ("nbrmask", "NbrMask ε-greedy")]:
        if key + "_curve" not in data:
            continue
        curve  = data[key + "_curve"]
        pA_val = data["phase_A"].get(key, float("nan"))
        pB_val = data["phase_B"].get(key, float("nan"))
        spike  = curve[PHASE_A_EPS]
        print(f"  {label:<28} {pA_val:>10.3f} {pB_val:>10.3f} {spike:>14.3f}")
    ospf_pA = data["phase_A"]["ospf"]; ospf_pB = data["phase_B"]["ospf"]
    print(f"  {'OSPF':<28} {ospf_pA:>10.3f} {ospf_pB:>10.3f} {'—':>14}")
    print(f"{'─'*60}")

    replot(data)
    print("Done.")


if __name__ == "__main__":
    main()
