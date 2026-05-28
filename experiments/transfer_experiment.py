#!/usr/bin/env python3
"""
Transfer-learning experiment: Abilene UCB → Germany50.

Takes one node's trained weights from a ScaIR (UCB) checkpoint trained on a
source topology and copies them to ALL agents on Germany50.  The Q-network
first layer is zero-padded to account for the larger destination one-hot
(max_nodes 30 → 50); all other layers have identical shapes and are copied
directly.  The source GNN state buffer V is discarded — agents reset their
own V to one_hot(node_id) at episode start anyway.

For each D_r the script runs two 200-episode training curves in parallel:
  (A) Germany50 UCB from scratch (random init)
  (B) Germany50 UCB starting from transferred Abilene weights

Outputs:
  <results>/transfer_curves.png   -- training curves A vs B per D_r
  <results>/transfer_results.json -- final eval stats
"""

import argparse
import json
import os
import random
import sys
import time
from typing import List

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source_checkpoint", required=True,
                   help="Checkpoint file from the source topology training")
    p.add_argument("--source_topo", required=True,
                   help="Topology file used when training the source checkpoint")
    p.add_argument("--source_node", type=int, default=0,
                   help="Which node's weights to copy (default: 0)")
    p.add_argument("--topo",   default="data/GER50/Topology.txt")
    p.add_argument("--tm_dir", default="data/GER50/TrafficMatrix")
    p.add_argument("--results", default="results/06_transfer")
    p.add_argument("--episodes",      type=int,   default=200)
    p.add_argument("--eval_episodes", type=int,   default=50)
    p.add_argument("--packets",       type=int,   default=100)
    p.add_argument("--dr_values",     type=float, nargs="+",
                   default=[0.0, 0.4, 0.8])
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--log_interval",  type=int,   default=20)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Weight transfer
# ---------------------------------------------------------------------------

def transfer_weights(ckpt_path: str, src_node: int, agents: list,
                     src_max_nodes: int, tgt_max_nodes: int,
                     max_degree: int, feature_length: int,
                     action_history_len: int) -> None:
    """
    Copy one node's Q-net + GNN weights from a checkpoint to every agent.

    Q-net first layer layout: [dest_oh | queues | feature_vec | action_hist]
    The destination one-hot block grows from src_max_nodes to tgt_max_nodes;
    the extra columns are zero-initialised (no prior knowledge of new destinations).
    All other blocks are identical in size and copied directly.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src_state = ckpt[str(src_node)]

    # Slice boundaries in the source first layer
    K, F, A = max_degree, feature_length, action_history_len * max_degree
    src_dest_end = src_max_nodes
    src_q_end    = src_dest_end + K
    src_fv_end   = src_q_end + F
    # src_act_end  = src_fv_end + A  (= total src input dim)

    # Slice boundaries in the target first layer
    tgt_dest_end = tgt_max_nodes
    tgt_q_end    = tgt_dest_end + K
    tgt_fv_end   = tgt_q_end + F
    # tgt_act_end  = tgt_fv_end + A  (= total tgt input dim)

    src_w0 = src_state["q_net"]["net.0.weight"]      # [N_u, src_in]
    n_units = src_w0.shape[0]

    for agent in agents:
        # --- SubGNN: f_w and g_w have the same shape regardless of topology ---
        sub_sd = agent.sub_gnn.state_dict()
        for k, v in src_state["sub_gnn"].items():
            if k == "V":
                continue          # V is reset to one_hot(node_id) each episode
            if k in sub_sd:
                sub_sd[k] = v.clone()
        agent.sub_gnn.load_state_dict(sub_sd)

        # --- QNetwork first layer: zero-pad destination block ---
        tgt_w0 = torch.zeros(n_units, tgt_dest_end + K + F + A)
        tgt_w0[:, :src_dest_end]               = src_w0[:, :src_dest_end]      # known dest (0..29)
        # columns tgt_dest_end:tgt_dest_end+K stay zero                        # new dest (30..49)
        tgt_w0[:, tgt_dest_end:tgt_q_end]      = src_w0[:, src_dest_end:src_q_end]   # queues
        tgt_w0[:, tgt_q_end:tgt_fv_end]        = src_w0[:, src_q_end:src_fv_end]     # feature vec
        tgt_w0[:, tgt_fv_end:]                 = src_w0[:, src_fv_end:]               # action hist

        q_sd = agent.q_net.state_dict()
        q_sd["net.0.weight"] = tgt_w0
        q_sd["net.0.bias"]   = src_state["q_net"]["net.0.bias"].clone()
        q_sd["net.2.weight"] = src_state["q_net"]["net.2.weight"].clone()
        q_sd["net.2.bias"]   = src_state["q_net"]["net.2.bias"].clone()
        q_sd["net.4.weight"] = src_state["q_net"]["net.4.weight"].clone()
        q_sd["net.4.bias"]   = src_state["q_net"]["net.4.bias"].clone()
        agent.q_net.load_state_dict(q_sd)
        agent.q_net_target.load_state_dict(q_sd)   # sync target


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(agents, env, tms, cfg, n_episodes, label, log_interval):
    curve = []
    t0 = time.time()
    for ep in range(1, n_episodes + 1):
        tm   = tms[(ep - 1) % len(tms)]
        pkts = env.generate_packets(tm, cfg.packets_per_episode)
        stats = env.run_episode(pkts, agents, training=True)
        curve.append(stats["avg_delivery_time"])

        if ep == 10:
            for ag in agents:
                ag.set_learning_rate(cfg.learning_rate)
        if ep % cfg.target_update_freq == 0:
            for ag in agents:
                ag.update_target()
        if ep % cfg.sigma_decay_freq == 0:
            for ag in agents:
                ag.decay_sigma()

        if ep % log_interval == 0 or ep == 1:
            elapsed = time.time() - t0
            print(f"  [{label}] ep {ep:3d}/{n_episodes}  "
                  f"time={stats['avg_delivery_time']:.2f}ms  ({elapsed:.0f}s)")
    return curve


def run_eval(agents, env, tms, cfg, n_episodes, offset):
    times = []
    for ep in range(1, n_episodes + 1):
        tm   = tms[(offset + ep - 1) % len(tms)]
        pkts = env.generate_packets(tm, cfg.packets_per_episode)
        stats = env.run_episode(pkts, agents, training=False)
        times.append(stats["avg_delivery_time"])
    return float(np.mean(times))


# ---------------------------------------------------------------------------
# Smooth helper
# ---------------------------------------------------------------------------

def smooth(values, w=10):
    out = []
    for i in range(len(values)):
        lo = max(0, i - w // 2)
        hi = min(len(values), i + w // 2 + 1)
        out.append(float(np.mean(values[lo:hi])))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.results, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- topologies ----
    src_topo = load_topology(args.source_topo)
    src_cfg  = ScaIRConfig()
    src_max_deg = max(len(v) for v in src_topo.adjacency.values())
    if src_topo.num_nodes > src_cfg.max_nodes: src_cfg.max_nodes = src_topo.num_nodes
    if src_max_deg > src_cfg.max_degree:       src_cfg.max_degree = src_max_deg
    src_max_nodes = src_cfg.max_nodes

    topo = load_topology(args.topo)
    raw_tms = load_all_traffic_matrices(args.tm_dir, topo.num_nodes)
    tms     = [normalise_tm(tm) for tm in raw_tms]
    print(f"Germany50: {topo.num_nodes} nodes, {len(tms)} TMs")

    base_cfg = ScaIRConfig()
    max_deg  = max(len(v) for v in topo.adjacency.values())
    if topo.num_nodes > base_cfg.max_nodes: base_cfg.max_nodes = topo.num_nodes
    if max_deg > base_cfg.max_degree:       base_cfg.max_degree = max_deg
    base_cfg.action_method = "ucb"
    base_cfg.packets_per_episode = args.packets
    tgt_max_nodes = base_cfg.max_nodes

    print(f"\nSource topology: {src_topo.num_nodes} nodes, max_nodes={src_max_nodes}")
    print(f"Target topology: {topo.num_nodes} nodes, max_nodes={tgt_max_nodes}")
    print(f"Q-net first layer: {src_max_nodes}+10+128+50={src_max_nodes+188} → "
          f"{tgt_max_nodes}+10+128+50={tgt_max_nodes+188}")
    print(f"  Destination block: {src_max_nodes} cols copied, "
          f"{tgt_max_nodes-src_max_nodes} cols zero-initialised")

    all_results = {}

    for dr in args.dr_values:
        import copy
        cfg = copy.copy(base_cfg)
        cfg.distribution_ratio = dr

        print(f"\n{'='*65}")
        print(f"  D_r = {dr}")
        print(f"{'='*65}")

        # ---- (A) Fresh training ----
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        agents_fresh = build_agents(topo, cfg)
        env = RoutingEnvironment(topo, cfg)
        print("  (A) Fresh Germany50 UCB...")
        curve_fresh = run_training(agents_fresh, env, tms, cfg,
                                   args.episodes, f"D_r={dr} Fresh", args.log_interval)
        eval_fresh  = run_eval(agents_fresh, env, tms, cfg, args.eval_episodes, args.episodes)
        print(f"    Fresh eval: {eval_fresh:.3f} ms")

        # ---- (B) Transfer from source checkpoint ----
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        agents_transfer = build_agents(topo, cfg)
        transfer_weights(
            ckpt_path=args.source_checkpoint,
            src_node=args.source_node,
            agents=agents_transfer,
            src_max_nodes=src_max_nodes,
            tgt_max_nodes=tgt_max_nodes,
            max_degree=base_cfg.max_degree,
            feature_length=base_cfg.feature_length,
            action_history_len=base_cfg.action_history_len,
        )
        env2 = RoutingEnvironment(topo, cfg)
        print(f"  (B) Transfer from node {args.source_node} of {args.source_checkpoint}...")
        curve_transfer = run_training(agents_transfer, env2, tms, cfg,
                                      args.episodes, f"D_r={dr} Transfer", args.log_interval)
        eval_transfer  = run_eval(agents_transfer, env2, tms, cfg, args.eval_episodes, args.episodes)
        print(f"    Transfer eval: {eval_transfer:.3f} ms")

        all_results[dr] = {
            "curve_fresh":    curve_fresh,
            "curve_transfer": curve_transfer,
            "eval_fresh":     eval_fresh,
            "eval_transfer":  eval_transfer,
        }

    # ---- Plot ----
    n = len(args.dr_values)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    fig.suptitle(
        f"Transfer Learning: Abilene UCB (node {args.source_node}) → Germany50\n"
        f"All 50 nodes initialised from source; fine-tuned with UCB",
        fontsize=12,
    )
    if n == 1:
        axes = [axes]

    for ax, dr in zip(axes, args.dr_values):
        r = all_results[dr]
        eps = list(range(1, args.episodes + 1))
        ax.plot(eps, smooth(r["curve_fresh"]),    color="steelblue",  linewidth=2,
                label=f"Fresh (eval {r['eval_fresh']:.2f} ms)")
        ax.plot(eps, smooth(r["curve_transfer"]), color="darkorange", linewidth=2,
                label=f"Transfer (eval {r['eval_transfer']:.2f} ms)")
        ax.set_title(f"D_r = {dr}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Delivery Time (ms)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(args.results, "transfer_curves.png")
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_png}")

    # ---- JSON ----
    json_out = {}
    for dr, r in all_results.items():
        json_out[str(dr)] = {
            "eval_fresh":    r["eval_fresh"],
            "eval_transfer": r["eval_transfer"],
        }
    json_path = os.path.join(args.results, "transfer_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "source_checkpoint": args.source_checkpoint,
            "source_node": args.source_node,
            "episodes": args.episodes,
            "results": json_out,
        }, f, indent=2)
    print(f"Saved: {json_path}")

    # ---- Summary ----
    print(f"\n{'D_r':>5}  {'Fresh eval':>12}  {'Transfer eval':>14}  {'Delta':>8}")
    print("-" * 50)
    for dr in args.dr_values:
        r = all_results[dr]
        delta = r["eval_transfer"] - r["eval_fresh"]
        sign = "+" if delta > 0 else ""
        print(f"{dr:>5.1f}  {r['eval_fresh']:>12.3f}  {r['eval_transfer']:>14.3f}  "
              f"{sign}{delta:.3f} ms")


if __name__ == "__main__":
    main()
