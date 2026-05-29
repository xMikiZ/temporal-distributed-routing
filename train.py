"""
ScaIR training script.

Usage examples
--------------
# Train on Abilene with defaults:
python train.py --topo data/Abi/Topology.txt --tm_dir "data/Abi/traffic matrix"

# Train with custom hyperparameters:
python train.py --topo data/Abi/Topology.txt --tm_dir "data/Abi/traffic matrix" \\
    --feature_length 64 --packets 100 --episodes 200

# Save checkpoints every 50 episodes to ./runs/exp1:
python train.py --topo data/Abi/Topology.txt --tm_dir "data/Abi/traffic matrix" \\
    --save_dir runs/exp1

Run `python train.py --help` for all options.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

from scair.agent import IRrAgent
from scair.config import ScaIRConfig
from scair.data_loader import load_all_traffic_matrices, load_topology, normalise_tm
from scair.environment import RoutingEnvironment


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ScaIR routing agents")

    # Data
    p.add_argument("--topo", required=True,
                   help="Path to topology file (Abilene format)")
    p.add_argument("--tm_dir", required=True,
                   help="Directory containing traffic-matrix .dat files")
    p.add_argument("--link_weights", default=None,
                   help="Optional link_weight.json for per-link delays")

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
                   help="Hot-spot traffic ratio D_r (default 0.0)")

    # Action selection
    p.add_argument("--action_method", type=str, default=None,
                   choices=["epsilon_greedy", "ucb"],
                   help="Action selection method (default: epsilon_greedy)")
    p.add_argument("--ucb_c", type=float, default=None,
                   help="UCB exploration constant c (default 2.0)")

    # Misc
    p.add_argument("--save_dir", type=str, default="checkpoints",
                   help="Directory to save model checkpoints")
    p.add_argument("--save_freq", type=int, default=400,
                   help="Save checkpoint every N episodes (0 = only at end)")
    p.add_argument("--resume", type=str, default=None, metavar="CHECKPOINT",
                   help="Resume training from this checkpoint file "
                        "(episode number is inferred from the filename)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--log_interval", type=int, default=10,
                   help="Print stats every N episodes")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_agents(topo, cfg: ScaIRConfig, gnn_cls=None) -> list:
    """Per-node SubGNN agents.  Pass gnn_cls=AttentionSubGNN for attention variant."""
    return [
        IRrAgent(
            node_id=n,
            neighbours=topo.adjacency[n],
            num_nodes=topo.num_nodes,
            cfg=cfg,
            gnn_cls=gnn_cls,
        )
        for n in range(topo.num_nodes)
    ]


def build_agents_topo_init(topo, cfg: ScaIRConfig, gnn_cls, init_vs: dict) -> list:
    """Per-node GNN agents with topology-derived V initialisation.

    init_vs: dict {node_id: torch.Tensor} — one init vector per node.
    gnn_cls must accept init_v keyword (PaperSubGNN, DotAttnSubGNN, LearnableAttnSubGNN).
    """
    return [
        IRrAgent(
            node_id=n,
            neighbours=topo.adjacency[n],
            num_nodes=topo.num_nodes,
            cfg=cfg,
            gnn_cls=gnn_cls,
            init_v=init_vs[n],
        )
        for n in range(topo.num_nodes)
    ]


def build_agents_fixed_topo(topo, cfg: ScaIRConfig, init_vs: dict) -> list:
    """Fixed (no-GNN) agents with topology-derived constant feature vectors.

    init_vs: dict {node_id: torch.Tensor}.
    """
    from scair.models import make_fixed_gnn
    cls = make_fixed_gnn(init_vs)
    return build_agents_no_gnn(topo, cfg, cls)


def build_agents_shared_gnn(topo, cfg: ScaIRConfig, gnn_cls=None) -> list:
    """All agents share f_w / g_w weights but each keeps its own V state.

    A template SubGNN owns the weight tensors; every agent receives a
    SharedWeightsSubGNN wrapper that points to those same tensors but
    holds a separate V buffer.  This is the correct "shared weights"
    interpretation: topology-aware feature vectors differ per node while
    the aggregation / output networks are jointly trained.

    Every agent accumulates gradients into the shared f_w / g_w during
    train_step(); call agents[0].shared_gnn_step(len(agents)) once per
    training episode to average and apply those gradients.

    Pass gnn_cls=AttentionSubGNN for the attention-aggregation variant.
    """
    from scair.models import SubGNN, AttentionSubGNN, make_shared_node_gnn
    cls = gnn_cls or SubGNN
    template = cls(0, topo.num_nodes, cfg.feature_length, cfg.neural_units)
    shared_gnn_opt = torch.optim.RMSprop(
        template.parameters(), lr=cfg.gnn_learning_rate
    )
    agents = []
    for n in range(topo.num_nodes):
        node_gnn = make_shared_node_gnn(n, cfg.feature_length, template)
        agent = IRrAgent(
            node_id=n,
            neighbours=topo.adjacency[n],
            num_nodes=topo.num_nodes,
            cfg=cfg,
            shared_sub_gnn=node_gnn,
            shared_gnn_opt=shared_gnn_opt if n == 0 else None,
        )
        agents.append(agent)
    return agents


def build_agents_no_gnn(topo, cfg: ScaIRConfig, gnn_cls) -> list:
    """Per-node Q-nets with a fixed (non-trained) feature vector instead of SubGNN.

    gnn_cls must accept (node_id, neighbors, feature_length) — i.e.
    OneHotSubGNN or NeighborMaskSubGNN.  The fixed GNN is passed as
    shared_sub_gnn so each agent's optimizer covers only its Q-net.
    """
    return [
        IRrAgent(
            node_id=n,
            neighbours=topo.adjacency[n],
            num_nodes=topo.num_nodes,
            cfg=cfg,
            shared_sub_gnn=gnn_cls(n, topo.adjacency[n], cfg.feature_length),
            shared_gnn_opt=None,
        )
        for n in range(topo.num_nodes)
    ]


def build_agents_no_gnn_shared_q(topo, cfg: ScaIRConfig, gnn_cls) -> list:
    """Shared Q-net with a fixed feature vector — no GNN at all.

    All agents share one QNetwork and one optimizer.  Each agent makes
    independent gradient updates to the shared Q-net (zero_grad → backward →
    step per agent per learning_cycle), giving the shared network N× more
    updates than a per-node Q-net.  This tests whether a single Q-net can
    generalise across all node positions given only topology-encoded inputs.
    """
    from scair.models import QNetwork

    shared_q = QNetwork(
        max_nodes=cfg.max_nodes,
        max_degree=cfg.max_degree,
        feature_length=cfg.feature_length,
        neural_units=cfg.neural_units,
        action_history_len=cfg.action_history_len,
    )
    shared_target = QNetwork(
        max_nodes=cfg.max_nodes,
        max_degree=cfg.max_degree,
        feature_length=cfg.feature_length,
        neural_units=cfg.neural_units,
        action_history_len=cfg.action_history_len,
    )
    shared_target.load_state_dict(shared_q.state_dict())
    for p in shared_target.parameters():
        p.requires_grad_(False)
    shared_q_opt = torch.optim.RMSprop(
        shared_q.parameters(), lr=cfg.learning_rate_initial
    )

    agents = build_agents_no_gnn(topo, cfg, gnn_cls)
    for agent in agents:
        agent.q_net = shared_q
        agent.q_net_target = shared_target
        agent.optimizer = shared_q_opt   # all agents share the same optimizer object
    return agents


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
        # Infer the episode number from the filename (e.g. episode_0400.pt -> 400)
        basename = os.path.splitext(os.path.basename(args.resume))[0]
        try:
            start_ep = int(basename.split("_")[-1]) + 1
        except ValueError:
            start_ep = 1
        print(f"  Resuming from episode {start_ep}")
        # Load existing history so the final history.json stays continuous
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

        # Cycle through traffic matrices
        tm = tms[(ep - 1) % len(tms)]

        # Generate packets
        packets = env.generate_packets(tm, cfg.packets_per_episode)

        # Run one episode (Algorithm 1)
        stats = env.run_episode(packets, agents, training=True)
        history.append(stats)

        # ---- learning-rate schedule (paper §5.1: L_r=0.1 for ep<=10, 0.001 after) ----
        if ep == 10:
            for agent in agents:
                agent.set_learning_rate(cfg.learning_rate)

        # ---- target network update every target_update_freq episodes ----
        if ep % cfg.target_update_freq == 0:
            for agent in agents:
                agent.update_target()

        # ---- sigma decay every sigma_decay_freq episodes ----
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

        # ---- checkpoint ----
        if args.save_freq > 0 and ep % args.save_freq == 0:
            save_checkpoint(agents, ep, cfg.save_dir)

    # ---- final checkpoint ----
    last_ep = start_ep + remaining - 1
    save_checkpoint(agents, last_ep, cfg.save_dir)

    # ---- save training history ----
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
