"""
ScaIR evaluation script.

Compares ScaIR against three baselines:
  - OSPF      (Dijkstra shortest path, offline, hop-count metric)
  - Q-routing (tabular Q-learning, paper §2)
  - Random    (uniform random next-hop, sanity check)

Usage
-----
# Evaluate a trained checkpoint on Abilene:
python evaluate.py \\
    --topo data/Abi/Topology.txt \\
    --tm_dir "data/Abi/traffic matrix" \\
    --checkpoint checkpoints/episode_0400.pt

# Run without a checkpoint (random init) to see offline performance:
python evaluate.py --topo data/Abi/Topology.txt --tm_dir "data/Abi/traffic matrix"
"""

import argparse
import heapq
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from scair.agent import IRrAgent
from scair.config import ScaIRConfig
from scair.data_loader import load_all_traffic_matrices, load_topology, normalise_tm
from scair.environment import RoutingEnvironment, Topology
from train import build_agents, load_checkpoint


# ---------------------------------------------------------------------------
# OSPF baseline (Dijkstra, shortest-path by hop count)
# ---------------------------------------------------------------------------

def dijkstra(topo: Topology, source: int) -> Dict[int, List[int]]:
    """Return next-hop dict: dest -> next_hop_node (hop-count shortest path)."""
    INF = float("inf")
    dist = {n: INF for n in range(topo.num_nodes)}
    prev: Dict[int, Optional[int]] = {n: None for n in range(topo.num_nodes)}
    dist[source] = 0
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v in topo.adjacency[u]:
            nd = d + 1  # hop count
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    # Build next-hop table from source
    next_hop: Dict[int, int] = {}
    for dst in range(topo.num_nodes):
        if dst == source or dist[dst] == INF:
            continue
        node = dst
        while prev[node] != source:
            node = prev[node]
        next_hop[dst] = node
    return next_hop


def run_ospf(
    topo: Topology, cfg: ScaIRConfig, tm: np.ndarray, num_packets: int
) -> Dict:
    """
    Route packets using static OSPF (Dijkstra shortest path) through the same
    event-driven simulator as ScaIR, so queuing delays are accounted for.
    Routing decisions are fixed (shortest path); only the delivery-time
    measurement is what changes vs. the old hop-count-only version.
    """
    next_hops = {n: dijkstra(topo, n) for n in range(topo.num_nodes)}

    env = RoutingEnvironment(topo, cfg)
    packets = env.generate_packets(tm, num_packets)
    env.reset()

    heap: List = []
    for pkt in packets:
        heapq.heappush(heap, (pkt.birth_time, pkt.pid, pkt, pkt.source, None))

    delivery_times: List[float] = []
    hops_list: List[int] = []
    hop_counts: Dict[int, int] = defaultdict(int)
    queue_len: Dict[Tuple[int, int], int] = defaultdict(int)

    while heap:
        t, _, pkt, node, prev = heapq.heappop(heap)

        if prev is not None:
            queue_len[(prev, node)] = max(0, queue_len[(prev, node)] - 1)

        if node == pkt.destination:
            delivery_times.append(t - pkt.birth_time)
            hops_list.append(hop_counts[pkt.pid])
            continue

        if hop_counts[pkt.pid] >= cfg.max_hops:
            continue

        next_node = next_hops.get(node, {}).get(pkt.destination)
        if next_node is None:
            continue

        q = queue_len[(node, next_node)]
        cost = q * cfg.queue_time_per_packet + cfg.transmission_time
        queue_len[(node, next_node)] += 1
        hop_counts[pkt.pid] += 1

        heapq.heappush(heap, (t + cost, pkt.pid, pkt, next_node, node))

    return {
        "avg_delivery_time": float(np.mean(delivery_times)) if delivery_times else float("inf"),
        "avg_hops": float(np.mean(hops_list)) if hops_list else 0.0,
        "delivered": len(delivery_times),
    }


# ---------------------------------------------------------------------------
# Q-routing baseline (tabular, paper §2)
# ---------------------------------------------------------------------------

class QRoutingAgent:
    """
    Classic Q-routing (Boyan & Littman, 1993).
    Q[dest][neighbour] = estimated delivery time.
    """

    def __init__(self, node_id: int, neighbours: List[int], num_nodes: int,
                 alpha: float = 0.1) -> None:
        self.node_id = node_id
        self.neighbours = neighbours
        self.alpha = alpha
        # Q-table: destination × neighbour
        self.Q: Dict[int, Dict[int, float]] = {
            d: {nb: 0.0 for nb in neighbours} for d in range(num_nodes)
        }

    def select_action(self, dest: int) -> Tuple[int, int]:
        q_row = self.Q[dest]
        best_nb = min(q_row, key=q_row.get)
        return self.neighbours.index(best_nb), best_nb

    def update(self, dest: int, nb: int, cost: float, nb_min_q: float) -> None:
        self.Q[dest][nb] += self.alpha * (cost + nb_min_q - self.Q[dest][nb])

    def min_q(self, dest: int) -> float:
        return min(self.Q[dest].values()) if self.Q[dest] else 0.0


def run_q_routing(
    topo: Topology, cfg: ScaIRConfig, tm: np.ndarray, num_packets: int, episodes: int = 200
) -> Dict:
    """Run tabular Q-routing for `episodes` training episodes, report the last one."""
    agents = [
        QRoutingAgent(n, topo.adjacency[n], topo.num_nodes)
        for n in range(topo.num_nodes)
    ]
    env = RoutingEnvironment(topo, cfg)
    last_stats = {"avg_delivery_time": float("inf"), "avg_hops": 0.0, "delivered": 0}

    for ep in range(episodes):
        packets = env.generate_packets(tm, num_packets)
        env.reset()

        # Simple event-driven loop (same logic as RoutingEnvironment but with Q-routing)
        heap = []
        for pkt in packets:
            heapq.heappush(heap, (pkt.birth_time, pkt.pid, pkt, pkt.source, None))

        delivery_times, hops_list = [], []
        hop_counts: Dict[int, int] = defaultdict(int)
        queue_len: Dict[Tuple[int, int], int] = defaultdict(int)

        while heap:
            t, _, pkt, node, prev = heapq.heappop(heap)

            if prev is not None:
                queue_len[(prev, node)] = max(0, queue_len[(prev, node)] - 1)

            if node == pkt.destination:
                delivery_times.append(t - pkt.birth_time)
                hops_list.append(hop_counts[pkt.pid])
                continue

            agent = agents[node]
            _, next_node = agent.select_action(pkt.destination)

            q = queue_len[(node, next_node)]
            cost = q * cfg.queue_time_per_packet + cfg.transmission_time
            queue_len[(node, next_node)] += 1
            hop_counts[pkt.pid] += 1

            nb_min = agents[next_node].min_q(pkt.destination)
            agent.update(pkt.destination, next_node, cost, nb_min)

            heapq.heappush(heap, (t + cost, pkt.pid, pkt, next_node, node))

        last_stats = {
            "avg_delivery_time": float(np.mean(delivery_times)) if delivery_times else float("inf"),
            "avg_hops": float(np.mean(hops_list)) if hops_list else 0.0,
            "delivered": len(delivery_times),
        }

    return last_stats


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

def plot_history(history_path: str, save_path: Optional[str] = None) -> None:
    """
    Load a history.json produced by train.py and display training evolution.

    Shows three subplots:
      - Average packet delivery time per episode
      - Average hops per episode
      - Average training loss per episode

    Requires matplotlib.  Falls back to a text summary if not installed.
    """
    if not os.path.exists(history_path):
        print(f"History file not found: {history_path}")
        return

    with open(history_path) as f:
        data = json.load(f)

    episodes = data["episodes"]
    stats = data["stats"]

    times  = [s["avg_delivery_time"] for s in stats]
    hops   = [s["avg_hops"]          for s in stats]
    losses = [s["avg_loss"]          for s in stats]

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        fig.suptitle("ScaIR Training Evolution", fontsize=13)

        axes[0].plot(episodes, times, linewidth=0.8)
        axes[0].set_ylabel("Avg delivery time (ms)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(episodes, hops, linewidth=0.8, color="tab:orange")
        axes[1].set_ylabel("Avg hops")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(episodes, losses, linewidth=0.8, color="tab:red")
        axes[2].set_ylabel("Avg training loss")
        axes[2].set_xlabel("Episode")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    except ImportError:
        # Text fallback when matplotlib is not available
        print("\n--- Training history (text summary) ---")
        print(f"{'Episode':>8}  {'Delivery (ms)':>14}  {'Hops':>6}  {'Loss':>12}")
        print("-" * 48)
        step = max(1, len(episodes) // 20)   # ~20 rows
        for i in range(0, len(episodes), step):
            print(f"{episodes[i]:>8}  {times[i]:>14.3f}  {hops[i]:>6.2f}  {losses[i]:>12.4f}")
        # Always show last episode
        if (len(episodes) - 1) % step != 0:
            i = len(episodes) - 1
            print(f"{episodes[i]:>8}  {times[i]:>14.3f}  {hops[i]:>6.2f}  {losses[i]:>12.4f}")
        print()
        print(f"Best delivery time: {min(times):.3f} ms  (episode {episodes[times.index(min(times))]})")
        print(f"Final delivery time: {times[-1]:.3f} ms")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ScaIR and baselines")
    p.add_argument("--topo", required=True)
    p.add_argument("--tm_dir", required=True)
    p.add_argument("--link_weights", default=None)
    p.add_argument("--checkpoint", default=None,
                   help="Path to a .pt checkpoint for ScaIR (optional)")
    p.add_argument("--history", default=None, metavar="HISTORY_JSON",
                   help="Path to history.json from train.py to plot training evolution. "
                        "If this is the only goal, pass --topo and --tm_dir as placeholders "
                        "and the evaluation section will be skipped.")
    p.add_argument("--save_plot", default=None, metavar="FILE",
                   help="Save the history plot to this file instead of showing it "
                        "(e.g. training.png).  Requires --history.")
    p.add_argument("--skip_eval", action="store_true",
                   help="Only plot history, skip ScaIR/OSPF/Q-routing evaluation.")
    p.add_argument("--episodes", type=int, default=100,
                   help="Evaluation episodes (default 100)")
    p.add_argument("--packets", type=int, default=50)
    p.add_argument("--feature_length", type=int, default=128)
    p.add_argument("--neural_units", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def evaluate(args: argparse.Namespace) -> None:
    # ---- plot history if requested ----
    if args.history:
        plot_history(args.history, save_path=args.save_plot)
        if args.skip_eval:
            return

    if args.skip_eval:
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = ScaIRConfig(
        feature_length=args.feature_length,
        neural_units=args.neural_units,
        packets_per_episode=args.packets,
        max_episodes=args.episodes,
        sigma_initial=0.1,   # low exploration for evaluation
        sigma_min=0.1,
    )

    topo = load_topology(args.topo, link_weight_file=args.link_weights)
    print(f"Topology: {topo.num_nodes} nodes")

    max_deg = max(len(v) for v in topo.adjacency.values())
    if topo.num_nodes > cfg.max_nodes:
        cfg.max_nodes = topo.num_nodes
    if max_deg > cfg.max_degree:
        cfg.max_degree = max_deg

    raw_tms = load_all_traffic_matrices(args.tm_dir, topo.num_nodes)
    tms = [normalise_tm(tm) for tm in raw_tms]

    # ---------- ScaIR ----------
    agents = build_agents(topo, cfg)
    if args.checkpoint:
        load_checkpoint(agents, args.checkpoint)
    # Evaluate (no training)
    env = RoutingEnvironment(topo, cfg)
    scair_times, scair_hops = [], []
    for ep in range(args.episodes):
        tm = tms[ep % len(tms)]
        packets = env.generate_packets(tm, args.packets)
        stats = env.run_episode(packets, agents, training=False)
        scair_times.append(stats["avg_delivery_time"])
        scair_hops.append(stats["avg_hops"])

    # ---------- OSPF ----------
    ospf_times, ospf_hops = [], []
    for ep in range(args.episodes):
        tm = tms[ep % len(tms)]
        stats = run_ospf(topo, cfg, tm, args.packets)
        ospf_times.append(stats["avg_delivery_time"])
        ospf_hops.append(stats["avg_hops"])

    # ---------- Q-routing ----------
    tm_avg = tms[0]
    qr_stats = run_q_routing(topo, cfg, tm_avg, args.packets, episodes=args.episodes)

    # ---------- Report ----------
    print("\n" + "=" * 60)
    print(f"{'Method':<20} {'Avg Delivery (ms)':>20} {'Avg Hops':>12}")
    print("-" * 60)
    print(f"{'ScaIR':<20} {np.mean(scair_times):>20.3f} {np.mean(scair_hops):>12.2f}")
    print(f"{'OSPF':<20} {np.mean(ospf_times):>20.3f} {np.mean(ospf_hops):>12.2f}")
    print(f"{'Q-routing':<20} {qr_stats['avg_delivery_time']:>20.3f} "
          f"{qr_stats['avg_hops']:>12.2f}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate(parse_args())
