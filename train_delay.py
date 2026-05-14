"""
ScaIR training script with per-link propagation delays.

Identical to train.py but uses scair_delay, which replaces the fixed
transmission_time with the actual per-link delay from link_weight.json.
Pass --link_weights to activate real delays; without it the default 1.0 ms
fallback is used (same behaviour as train.py).

Usage examples
--------------
# Train on Abilene with real link delays:
python train_delay.py --topo data/ABI/Topology.txt --tm_dir data/ABI/TrafficMatrix \\
    --link_weights data/ABI/link_weight.json

# Train with custom hyperparameters:
python train_delay.py --topo data/ABI/Topology.txt --tm_dir data/ABI/TrafficMatrix \\
    --link_weights data/ABI/link_weight.json --feature_length 64 --packets 100 --episodes 200

Run `python train_delay.py --help` for all options.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

from scair_delay.agent import IRrAgent
from scair_delay.config import ScaIRConfig
from scair_delay.data_loader import load_all_traffic_matrices, load_topology, normalise_tm
from scair_delay.environment import RoutingEnvironment


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ScaIR routing agents (with link delays)")

    # Data
    p.add_argument("--topo", required=True,
                   help="Path to topology file (Abilene format)")
    p.add_argument("--tm_dir", required=True,
                   help="Directory containing traffic-matrix .dat files")
    p.add_argument("--link_weights", default=None,
                   help="link_weight.json with per-link delays (ms). "
                        "Without this, falls back to fixed 1.0 ms per hop.")

    # Training
    p.add_argument("--episodes", type=int, default=None,
                   help="Override max_episodes from config")
    p.add_argument("--packets", type=int, default=None,
                   help="Override packets_per_episode from config")

    # Sub-GNN
    p.add_argument("--feature_length", type=int, default=None,
                   help="Feature vector length F_l (default 128)")
    p.add_argument("--neural_units", type=int, default=None,
                   help="Hidden units N_u (default 64)")
    p.add_argument("--gnn_iters", type=int, default=None,
                   help="Sub-GNN periodic update iterations I_n (default 3)")

    # Traffic
    p.add_argument("--gen_interval", type=float, default=None,
                   help="Mean Poisson inter-arrival time G_i (default 0.5 ms)")
    p.add_argument("--dist_ratio", type=float, default=None,
                   help="Hot-spot traffic ratio D_r (default 0.5)")

    # Action selection
    p.add_argument("--action_method", type=str, default=None,
                   choices=["epsilon_greedy", "ucb"],
                   help="Action selection method (default: epsilon_greedy)")
    p.add_argument("--ucb_c", type=float, default=None,
                   help="UCB exploration constant c (default 2.0)")

    # Misc
    p.add_argument("--save_dir", type=str, default="checkpoints_delay",
                   help="Directory to save model checkpoints")
    p.add_argument("--save_freq", type=int, default=50,
                   help="Save checkpoint every N episodes (0 = only at end)")
    p.add_argument("--resume", type=str, default=None, metavar="CHECKPOINT",
                   help="Resume training from this checkpoint file")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--log_interval", type=int, default=10,
                   help="Print stats every N episodes")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_agents(topo, cfg: ScaIRConfig) -> list:
    return [
        IRrAgent(
            node_id=n,
            neighbours=topo.adjacency[n],
            num_nodes=topo.num_nodes,
            cfg=cfg,
        )
        for n in range(topo.num_nodes)
    ]


def save_checkpoint(agents: list, episode: int, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    state = {
        str(n): {
            "sub_gnn": agents[n].sub_gnn.state_dict(),
            "q_net": agents[n].q_net.state_dict(),
            "sigma": agents[n].sigma,
        }
        for n in range(len(agents))
    }
    path = os.path.join(save_dir, f"episode_{episode:04d}.pt")
    torch.save(state, path)
    print(f"  [checkpoint saved to {path}]")


def load_checkpoint(agents: list, path: str) -> None:
    state = torch.load(path, map_location="cpu")
    for n in range(len(agents)):
        key = str(n)
        if key in state:
            agents[n].sub_gnn.load_state_dict(state[key]["sub_gnn"])
            agents[n].q_net.load_state_dict(state[key]["q_net"])
            agents[n].sigma = state[key].get("sigma", agents[n].cfg.sigma_min)
    print(f"Loaded checkpoint from {path}")


# ---------------------------------------------------------------------------
# Training loop  (Algorithm 1)
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # ---- config ----
    cfg = ScaIRConfig()
    if args.episodes is not None:
        cfg.max_episodes = args.episodes
    if args.packets is not None:
        cfg.packets_per_episode = args.packets
    if args.feature_length is not None:
        cfg.feature_length = args.feature_length
    if args.neural_units is not None:
        cfg.neural_units = args.neural_units
    if args.gnn_iters is not None:
        cfg.gnn_update_iters = args.gnn_iters
    if args.gen_interval is not None:
        cfg.generation_interval = args.gen_interval
    if args.dist_ratio is not None:
        cfg.distribution_ratio = args.dist_ratio
    if args.action_method is not None:
        cfg.action_method = args.action_method
    if args.ucb_c is not None:
        cfg.ucb_c = args.ucb_c
    if args.save_dir:
        cfg.save_dir = args.save_dir
    if args.log_interval:
        cfg.log_interval = args.log_interval
    if args.seed is not None:
        cfg.seed = args.seed

    # ---- reproducibility ----
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    # ---- data ----
    print(f"Loading topology from {args.topo}")
    topo = load_topology(args.topo, link_weight_file=args.link_weights)
    if args.link_weights:
        print(f"  Link delays loaded from {args.link_weights}")
    else:
        print("  No link_weights provided — using fixed 1.0 ms per hop")
    print(f"  {topo.num_nodes} nodes, {sum(len(v) for v in topo.adjacency.values())//2} links")

    print(f"Loading traffic matrices from {args.tm_dir}")
    raw_tms = load_all_traffic_matrices(args.tm_dir, topo.num_nodes)
    tms = [normalise_tm(tm) for tm in raw_tms]
    print(f"  {len(tms)} traffic matrices loaded")

    # ---- check config dimensions ----
    max_deg = max(len(nbrs) for nbrs in topo.adjacency.values())
    if topo.num_nodes > cfg.max_nodes:
        cfg.max_nodes = topo.num_nodes
        print(f"  auto-adjusted max_nodes = {cfg.max_nodes}")
    if max_deg > cfg.max_degree:
        cfg.max_degree = max_deg
        print(f"  auto-adjusted max_degree = {cfg.max_degree}")

    # ---- agents ----
    agents = build_agents(topo, cfg)
    print(f"Created {len(agents)} IRr agents")

    # ---- resume from checkpoint ----
    start_ep = 1
    history = []
    if args.resume:
        load_checkpoint(agents, args.resume)
        basename = os.path.splitext(os.path.basename(args.resume))[0]
        try:
            start_ep = int(basename.split("_")[-1]) + 1
        except ValueError:
            start_ep = 1
        print(f"  Resuming from episode {start_ep}")
        history_path = os.path.join(cfg.save_dir, "history.json")
        if os.path.exists(history_path):
            with open(history_path) as f:
                saved = json.load(f)
            history = saved.get("stats", [])
            print(f"  Loaded {len(history)} existing history entries")

    # ---- environment ----
    env = RoutingEnvironment(topo, cfg)

    # ---- training loop ----
    remaining = cfg.max_episodes
    print(f"\nStarting training: episodes {start_ep}–{start_ep + remaining - 1}, "
          f"{cfg.packets_per_episode} packets/episode\n")

    for ep in range(start_ep, start_ep + remaining):
        t0 = time.time()

        tm = tms[(ep - 1) % len(tms)]
        packets = env.generate_packets(tm, cfg.packets_per_episode)
        stats = env.run_episode(packets, agents, training=True)
        history.append(stats)

        if ep == 10:
            for agent in agents:
                agent.set_learning_rate(cfg.learning_rate)

        if ep % cfg.target_update_freq == 0:
            for agent in agents:
                agent.update_target()

        if ep % cfg.sigma_decay_freq == 0:
            for agent in agents:
                agent.decay_sigma()

        elapsed = time.time() - t0

        if ep % cfg.log_interval == 0 or ep == 1:
            sigma = agents[0].sigma
            print(
                f"Episode {ep:4d}/{cfg.max_episodes}  "
                f"avg_time={stats['avg_delivery_time']:7.3f} ms  "
                f"avg_hops={stats['avg_hops']:5.2f}  "
                f"delivered={stats['delivered']:3d}/{cfg.packets_per_episode}  "
                f"loss={stats['avg_loss']:.4f}  "
                f"sigma={sigma:.2f}  "
                f"({elapsed:.1f}s)"
            )

        if args.save_freq > 0 and ep % args.save_freq == 0:
            save_checkpoint(agents, ep, cfg.save_dir)

    last_ep = start_ep + remaining - 1
    save_checkpoint(agents, last_ep, cfg.save_dir)

    history_path = os.path.join(cfg.save_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(
            {"episodes": list(range(1, len(history) + 1)), "stats": history},
            f,
            indent=2,
        )
    print(f"  [training history saved to {history_path}]")
    print("\nTraining complete.")

    return agents, history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
