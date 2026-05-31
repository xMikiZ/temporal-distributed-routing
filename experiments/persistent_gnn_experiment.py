#!/usr/bin/env python3
"""
Compare ScaIR with per-episode V_n reset (our default) against the paper's
one-time initialisation (V_n persists across episodes).

Current default
---------------
  start of every episode:
    sub_gnn.reset()          <- V_n back to one-hot
    _global_gnn_update(K=8)  <- 8 warm-up message-passing rounds

Paper-faithful (persistent GNN)
--------------------------------
  before episode 1 only:
    _global_gnn_update(K=8)  <- single warm-up at deployment
  every episode:
    skip reset, skip warm-up
    V_n continues to evolve via the existing I_n=3 periodic local updates

Both variants use identical seeds, traffic, and architecture on Germany50.
"""

import copy
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

RESULTS        = "results/persistent_gnn"
SEED           = 42
TOPO_FILE      = "data/GER50/Topology.txt"
TM_DIR         = "data/GER50/TrafficMatrix"
TRAIN_EPISODES = 300
EVAL_EPISODES  = 50
N_PACKETS      = 100
DR_VALUES      = [0.0, 0.4, 0.6, 0.8]
LOG_INTERVAL   = 50


# ---------------------------------------------------------------------------
# One-time global GNN warm-up (used before the persistent-GNN training loop)
# ---------------------------------------------------------------------------

def global_gnn_warmup(agents, topo, n_iters: int) -> None:
    """Run n_iters synchronous message-passing rounds across all agents."""
    for _ in range(n_iters):
        fvs = {n: agents[n].get_feature_vector() for n in range(topo.num_nodes)}
        for n in range(topo.num_nodes):
            nbr_fvs = [fvs[nb] for nb in topo.adjacency[n]]
            agents[n].sub_gnn.iterate(nbr_fvs)
            agents[n]._nbr_fvs = [fvs[nb].detach() for nb in topo.adjacency[n]]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_standard(topo, cfg, tms, n_episodes, n_packets, seed):
    """Standard ScaIR: V_n reset + K=8 warm-up at the start of every episode."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    agents = build_agents(topo, cfg)
    env = RoutingEnvironment(topo, cfg)  # uses cfg.gnn_init_iters = 8
    t0 = time.time()
    curve = []

    for ep in range(1, n_episodes + 1):
        tm = tms[(ep - 1) % len(tms)]
        packets = env.generate_packets(tm, n_packets)
        stats = env.run_episode(packets, agents, training=True)
        curve.append(stats["avg_delivery_time"])

        if ep == 10:
            for ag in agents: ag.set_learning_rate(cfg.learning_rate)
        if ep % cfg.target_update_freq == 0:
            for ag in agents: ag.update_target()
        if ep % cfg.sigma_decay_freq == 0:
            for ag in agents: ag.decay_sigma()
        if ep % LOG_INTERVAL == 0 or ep == 1:
            print(f"    [standard  ep {ep:3d}/{n_episodes}] "
                  f"t={stats['avg_delivery_time']:.2f}ms  "
                  f"σ={agents[0].sigma:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    return agents, curve


def train_persistent(topo, cfg, tms, n_episodes, n_packets, seed):
    """
    Persistent-GNN ScaIR: V_n is initialised once with K=8 rounds before
    training begins and never reset again.  Per-episode, only queues and
    action history are cleared; V_n continues to evolve through the
    I_n=3 periodic local GNN updates that happen during the episode itself.
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    agents = build_agents(topo, cfg)

    # One-time initialisation: K=8 global warm-up rounds
    global_gnn_warmup(agents, topo, n_iters=cfg.gnn_init_iters)

    # Disable V_n reset for all agents (no-op lambda)
    for ag in agents:
        ag.sub_gnn.reset = lambda: None

    # Use gnn_init_iters=0 so run_episode skips its own warm-up
    persistent_cfg = copy.copy(cfg)
    persistent_cfg.gnn_init_iters = 0
    env = RoutingEnvironment(topo, persistent_cfg)

    t0 = time.time()
    curve = []

    for ep in range(1, n_episodes + 1):
        tm = tms[(ep - 1) % len(tms)]
        packets = env.generate_packets(tm, n_packets)
        stats = env.run_episode(packets, agents, training=True)
        curve.append(stats["avg_delivery_time"])

        if ep == 10:
            for ag in agents: ag.set_learning_rate(cfg.learning_rate)
        if ep % persistent_cfg.target_update_freq == 0:
            for ag in agents: ag.update_target()
        if ep % persistent_cfg.sigma_decay_freq == 0:
            for ag in agents: ag.decay_sigma()
        if ep % LOG_INTERVAL == 0 or ep == 1:
            print(f"    [persistent ep {ep:3d}/{n_episodes}] "
                  f"t={stats['avg_delivery_time']:.2f}ms  "
                  f"σ={agents[0].sigma:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    return agents, curve, persistent_cfg


def eval_agents(agents, topo, cfg, eval_eps):
    env = RoutingEnvironment(topo, cfg)
    times = [env.run_episode(ep, agents, training=False)["avg_delivery_time"]
             for ep in eval_eps]
    return float(np.mean(times)), float(np.std(times))


def ospf_eval(topo, cfg, eval_eps):
    """Run OSPF on eval episodes (reuse optimal_comparison.run_ospf_episode)."""
    from experiments.optimal_comparison import run_ospf_episode
    times = [run_ospf_episode(topo, cfg, ep) for ep in eval_eps]
    return float(np.mean(times))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_results(results_by_dr, train_curves, out_dir):
    drs = sorted(results_by_dr.keys())

    # ----- training curves for DR=0.6 -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Standard vs Persistent GNN — Germany50", fontsize=13)

    ax = axes[0]
    win = 15
    for label, color in [("standard", "#e67e22"), ("persistent", "#3498db")]:
        raw = train_curves[label]
        sm  = np.convolve(raw, np.ones(win)/win, mode="valid")
        ax.plot(range(1, len(raw)+1), raw, color=color, alpha=0.2, linewidth=0.7)
        ax.plot(range(win, len(raw)+1), sm, color=color, linewidth=2, label=label.capitalize())

    # Reference lines from DR=0.6 results
    d = results_by_dr[0.6]
    ax.axhline(d["ospf"], color="red", linestyle="--", linewidth=1.2, label=f"OSPF {d['ospf']:.2f}ms")
    ax.set_xlabel("Training Episode"); ax.set_ylabel("Avg Delivery Time (ms)")
    ax.set_title("Training curve (D_r = 0.6)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ----- bar chart across DR values -----
    ax = axes[1]
    x = np.arange(len(drs))
    w = 0.25
    ospf_vals = [results_by_dr[dr]["ospf"] for dr in drs]
    std_vals  = [results_by_dr[dr]["standard"][0] for dr in drs]
    per_vals  = [results_by_dr[dr]["persistent"][0] for dr in drs]
    std_stds  = [results_by_dr[dr]["standard"][1] for dr in drs]
    per_stds  = [results_by_dr[dr]["persistent"][1] for dr in drs]

    ax.bar(x - w, ospf_vals, w, label="OSPF", color="#e74c3c", alpha=0.8)
    ax.bar(x,     std_vals,  w, label="Standard (reset)", color="#e67e22",
           alpha=0.85, yerr=std_stds, capsize=4)
    ax.bar(x + w, per_vals,  w, label="Persistent (no reset)", color="#3498db",
           alpha=0.85, yerr=per_stds, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([f"D_r={d}" for d in drs])
    ax.set_ylabel("Avg Delivery Time (ms)")
    ax.set_title("Eval results by D_r value")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "comparison.png")
    plt.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS, exist_ok=True)

    topo    = load_topology(TOPO_FILE)
    raw_tms = load_all_traffic_matrices(TM_DIR, topo.num_nodes)
    tms     = [normalise_tm(tm) for tm in raw_tms]
    print(f"Germany50: {topo.num_nodes} nodes, {len(tms)} TMs\n")

    results_by_dr  = {}
    # Store training curves only for DR=0.6 (representative)
    train_curves   = {}

    for dr in DR_VALUES:
        print(f"{'='*60}")
        print(f"  D_r = {dr}")
        print(f"{'='*60}", flush=True)

        cfg = ScaIRConfig()
        max_deg = max(len(v) for v in topo.adjacency.values())
        if topo.num_nodes > cfg.max_nodes:  cfg.max_nodes  = topo.num_nodes
        if max_deg        > cfg.max_degree: cfg.max_degree = max_deg
        cfg.action_method       = "ucb"
        cfg.distribution_ratio  = dr
        cfg.packets_per_episode = N_PACKETS

        # Pre-generate eval episodes (same for both variants)
        env_gen = RoutingEnvironment(topo, cfg)
        random.seed(SEED + 1000); np.random.seed(SEED + 1000)
        eval_eps = [env_gen.generate_packets(tms[i % len(tms)], N_PACKETS)
                    for i in range(EVAL_EPISODES)]

        ospf_val = ospf_eval(topo, cfg, eval_eps)
        print(f"  OSPF: {ospf_val:.3f} ms", flush=True)

        # --- Standard ---
        print(f"\n  [Standard] training ...", flush=True)
        random.seed(SEED); np.random.seed(SEED)
        std_agents, std_curve = train_standard(
            topo, cfg, tms, TRAIN_EPISODES, N_PACKETS, SEED)
        std_mean, std_std = eval_agents(std_agents, topo, cfg, eval_eps)
        print(f"  Standard  eval: {std_mean:.3f} ± {std_std:.3f} ms  "
              f"(+{(ospf_val-std_mean)/max(ospf_val,1e-6)*100:.1f}% vs OSPF)", flush=True)

        # --- Persistent ---
        print(f"\n  [Persistent] training ...", flush=True)
        random.seed(SEED); np.random.seed(SEED)
        per_agents, per_curve, per_cfg = train_persistent(
            topo, cfg, tms, TRAIN_EPISODES, N_PACKETS, SEED)
        per_mean, per_std = eval_agents(per_agents, topo, per_cfg, eval_eps)
        print(f"  Persistent eval: {per_mean:.3f} ± {per_std:.3f} ms  "
              f"(+{(ospf_val-per_mean)/max(ospf_val,1e-6)*100:.1f}% vs OSPF)", flush=True)

        delta = per_mean - std_mean
        print(f"\n  Δ (persistent − standard): {delta:+.3f} ms  "
              f"({'persistent worse' if delta>0 else 'persistent better'})", flush=True)

        results_by_dr[dr] = {
            "ospf":       ospf_val,
            "standard":   (std_mean, std_std),
            "persistent": (per_mean, per_std),
        }
        if abs(dr - 0.6) < 1e-9:
            train_curves["standard"]   = std_curve
            train_curves["persistent"] = per_curve

    # ----- Summary table -----
    print(f"\n{'='*60}")
    print("  SUMMARY — Germany50")
    print(f"{'='*60}")
    print(f"  {'D_r':<5}  {'OSPF':>8}  {'Standard':>10}  {'Persistent':>12}  {'Δ':>8}")
    for dr in DR_VALUES:
        d = results_by_dr[dr]
        delta = d["persistent"][0] - d["standard"][0]
        print(f"  {dr:<5.1f}  {d['ospf']:>8.3f}  "
              f"{d['standard'][0]:>10.3f}  {d['persistent'][0]:>12.3f}  "
              f"{delta:>+8.3f}")
    print(f"{'='*60}")

    # Save JSON
    out = {
        "topology": "germany50",
        "train_episodes": TRAIN_EPISODES,
        "results": {
            str(dr): {
                "ospf": v["ospf"],
                "standard": {"mean": v["standard"][0], "std": v["standard"][1]},
                "persistent": {"mean": v["persistent"][0], "std": v["persistent"][1]},
            }
            for dr, v in results_by_dr.items()
        },
        "training_curves_dr0.6": {
            "standard":   train_curves.get("standard",   []),
            "persistent": train_curves.get("persistent", []),
        },
    }
    json_path = os.path.join(RESULTS, "results.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {json_path}")

    plot_results(results_by_dr, train_curves, RESULTS)
    print("Done.")


if __name__ == "__main__":
    main()
