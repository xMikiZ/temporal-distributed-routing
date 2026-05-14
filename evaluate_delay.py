"""
ScaIR evaluation script with per-link propagation delays.

Identical to evaluate.py but uses scair_delay (real link delays in cost)
and weighted Dijkstra for OSPF (shortest delay path, not shortest hop path).

Pass --link_weights to load real delays; without it falls back to 1.0 ms/hop.

Usage
-----
# Evaluate a trained checkpoint on Abilene with real link delays:
python evaluate_delay.py \\
    --topo data/ABI/Topology.txt \\
    --tm_dir data/ABI/TrafficMatrix \\
    --link_weights data/ABI/link_weight.json \\
    --checkpoint checkpoints_delay/episode_0400.pt
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

from scair_delay.agent import IRrAgent
from scair_delay.config import ScaIRConfig
from scair_delay.data_loader import load_all_traffic_matrices, load_topology, normalise_tm
from scair_delay.environment import RoutingEnvironment, Topology
from train_delay import build_agents, load_checkpoint


# ---------------------------------------------------------------------------
# OSPF baseline (Dijkstra, shortest-path by propagation delay)
# ---------------------------------------------------------------------------

def dijkstra(topo: Topology, source: int) -> Dict[int, int]:
    """Return next-hop dict: dest -> next_hop_node (delay-weighted shortest path)."""
    INF = float("inf")
    dist = {n: INF for n in range(topo.num_nodes)}
    prev: Dict[int, Optional[int]] = {n: None for n in range(topo.num_nodes)}
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v in topo.adjacency[u]:
            nd = d + topo.delay(u, v)   # weighted by propagation delay
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

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
    Route packets using delay-weighted OSPF (Dijkstra on propagation delay).
    Queuing delays are still simulated by the event-driven loop.
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
        cost = q * cfg.queue_time_per_packet + topo.delay(node, next_node)
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
    def __init__(self, node_id: int, neighbours: List[int], num_nodes: int,
                 alpha: float = 0.1) -> None:
        self.node_id = node_id
        self.neighbours = neighbours
        self.alpha = alpha
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

            if hop_counts[pkt.pid] >= cfg.max_hops:
                continue

            agent = agents[node]
            _, next_node = agent.select_action(pkt.destination)

            q = queue_len[(node, next_node)]
            cost = q * cfg.queue_time_per_packet + topo.delay(node, next_node)
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
# Training history
# ---------------------------------------------------------------------------

def plot_history(history_path: str, save_path: Optional[str] = None) -> None:
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
        fig.suptitle("ScaIR Training Evolution (with link delays)", fontsize=13)

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
        print("\n--- Training history (text summary) ---")
        print(f"{'Episode':>8}  {'Delivery (ms)':>14}  {'Hops':>6}  {'Loss':>12}")
        print("-" * 48)
        step = max(1, len(episodes) // 20)
        for i in range(0, len(episodes), step):
            print(f"{episodes[i]:>8}  {times[i]:>14.3f}  {hops[i]:>6.2f}  {losses[i]:>12.4f}")
        if (len(episodes) - 1) % step != 0:
            i = len(episodes) - 1
            print(f"{episodes[i]:>8}  {times[i]:>14.3f}  {hops[i]:>6.2f}  {losses[i]:>12.4f}")
        print(f"\nBest delivery time: {min(times):.3f} ms  (episode {episodes[times.index(min(times))]})")
        print(f"Final delivery time: {times[-1]:.3f} ms")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ScaIR with link delays vs baselines")
    p.add_argument("--topo", required=True)
    p.add_argument("--tm_dir", required=True)
    p.add_argument("--link_weights", default=None,
                   help="link_weight.json with per-link delays (ms). "
                        "Without this, falls back to fixed 1.0 ms per hop.")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a .pt checkpoint for ScaIR (optional)")
    p.add_argument("--history", default=None, metavar="HISTORY_JSON",
                   help="Path to history.json from train_delay.py to plot training evolution.")
    p.add_argument("--save_plot", default=None, metavar="FILE",
                   help="Save the history plot to this file instead of showing it.")
    p.add_argument("--skip_eval", action="store_true",
                   help="Only plot history, skip ScaIR/OSPF/Q-routing evaluation.")
    p.add_argument("--episodes", type=int, default=100,
                   help="Evaluation episodes (default 100)")
    p.add_argument("--packets", type=int, default=50)
    p.add_argument("--dist_ratio", type=float, default=0.5,
                   help="Hot-spot traffic ratio D_r (default 0.5)")
    p.add_argument("--feature_length", type=int, default=128)
    p.add_argument("--neural_units", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def evaluate(args: argparse.Namespace) -> None:
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
        distribution_ratio=args.dist_ratio,
        sigma_initial=0.1,
        sigma_min=0.1,
    )

    topo = load_topology(args.topo, link_weight_file=args.link_weights)
    if args.link_weights:
        delays = list(topo.link_delays.values())
        print(f"Topology: {topo.num_nodes} nodes  |  "
              f"link delays: min={min(delays):.1f} max={max(delays):.1f} "
              f"mean={np.mean(delays):.1f} ms")
    else:
        print(f"Topology: {topo.num_nodes} nodes  |  link delays: fixed 1.0 ms (no link_weights)")

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
    env = RoutingEnvironment(topo, cfg)
    scair_times, scair_hops = [], []
    for ep in range(args.episodes):
        tm = tms[ep % len(tms)]
        packets = env.generate_packets(tm, args.packets)
        stats = env.run_episode(packets, agents, training=False)
        scair_times.append(stats["avg_delivery_time"])
        scair_hops.append(stats["avg_hops"])

    # ---------- OSPF (delay-weighted) ----------
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
    print(f"{'OSPF (delay)':<20} {np.mean(ospf_times):>20.3f} {np.mean(ospf_hops):>12.2f}")
    print(f"{'Q-routing':<20} {qr_stats['avg_delivery_time']:>20.3f} "
          f"{qr_stats['avg_hops']:>12.2f}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate(parse_args())
