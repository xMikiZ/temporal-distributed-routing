"""
Packet-routing simulation environment for ScaIR.

Implements Algorithm 1 (paper §4.3) in a Python event-driven simulator
that replaces OMNet++.  The simulator faithfully reproduces:

  - Poisson packet generation from a traffic matrix (paper §5.1)
  - Per-interface FIFO queues at each router                    (paper §3)
  - Cost model: c_t = q + h  (queuing time + 1 ms transmission)(paper §4.2)
  - σ-greedy action selection and DQN training                  (Alg. 1)
  - Sub-GNN periodic update triggered every gnn_update_period ticks

Two event types drive the simulation:
  ARRIVE  -- packet p arrives at node n (from node prev)
  GNN_UPDATE -- global GNN feature-vector exchange
"""

import heapq
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .agent import IRrAgent
from .config import ScaIRConfig


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

class Topology:
    """Lightweight network graph used by the environment."""

    def __init__(
        self,
        num_nodes: int,
        adjacency: Dict[int, List[int]],
        link_delays: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> None:
        self.num_nodes = num_nodes
        self.adjacency = adjacency                  # node -> [neighbours]
        self.link_delays = link_delays or {}        # (u,v) -> ms

    def delay(self, u: int, v: int, default: float = 1.0) -> float:
        return self.link_delays.get((u, v), default)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _Event:
    time: float
    kind: str = field(compare=False)
    data: dict = field(compare=False, default_factory=dict)


# ---------------------------------------------------------------------------
# Packet
# ---------------------------------------------------------------------------

@dataclass
class Packet:
    pid: int
    source: int
    destination: int
    birth_time: float


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class RoutingEnvironment:
    """
    Event-driven routing simulation.

    Usage per episode
    -----------------
    env.reset()
    packets = env.generate_packets(traffic_matrix, num_packets)
    stats   = env.run_episode(packets, agents, training=True)
    """

    def __init__(self, topology: Topology, cfg: ScaIRConfig) -> None:
        self.topo = topology
        self.cfg = cfg
        self._heap: List[_Event] = []
        self.now: float = 0.0

    # ------------------------------------------------------------------
    # Packet generation (paper §5.1)
    # ------------------------------------------------------------------

    def generate_packets(
        self,
        traffic_matrix: np.ndarray,
        num_packets: int,
    ) -> List[Packet]:
        """
        Generate `num_packets` packets with Poisson inter-arrivals.

        Source-destination pairs are sampled proportional to the traffic
        matrix, with D_r fraction forced onto the hot-spot path
        (node 0 -> last node).

        The paper: "actual arrival of traffic is subject to Poisson distribution."
        G_i is used as the mean inter-arrival time between consecutive packets.
        """
        n = self.topo.num_nodes
        tm = traffic_matrix[:n, :n].copy().astype(float)
        np.fill_diagonal(tm, 0.0)

        total = tm.sum()
        if total > 0:
            probs = (tm / total).flatten()
        else:
            probs = np.ones(n * n) / max(n * n - n, 1)
            for i in range(n):
                probs[i * n + i] = 0.0

        last_node = n - 1
        packets: List[Packet] = []
        t = 0.0
        for pid in range(num_packets):
            t += np.random.exponential(self.cfg.generation_interval)
            if random.random() < self.cfg.distribution_ratio:
                src, dst = 0, last_node
            else:
                idx = int(np.random.choice(n * n, p=probs))
                src, dst = divmod(idx, n)
            packets.append(Packet(pid, src, dst, t))

        return packets

    # ------------------------------------------------------------------
    # Episode
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._heap = []
        self.now = 0.0

    def run_episode(
        self,
        packets: List[Packet],
        agents: List[IRrAgent],
        training: bool = True,
    ) -> Dict:
        """
        Route all packets from source to destination.

        Implements Algorithm 1:
          - Each packet arrival triggers a routing decision (decision tick).
          - Every learning_cycle ticks: gradient-descent update.
          - Every gnn_update_period ticks: global GNN message-passing round.

        Returns a stats dict:
            avg_delivery_time  -- primary QoS metric (paper Eq. 1)
            avg_hops           -- secondary metric
            delivered          -- number of successfully delivered packets
            avg_loss           -- mean training loss (0 if not training)
        """
        self.reset()
        for agent in agents:
            agent.reset_episode()

        # Initialise GNNs with K iterations (paper §3, init stage)
        self._global_gnn_update(agents, n_iters=self.cfg.gnn_init_iters)

        # Schedule initial arrivals at source nodes
        for pkt in packets:
            self._push(pkt.birth_time, "arrive", {"pkt": pkt, "node": pkt.source, "prev": None})

        delivery_times: List[float] = []
        hops_list: List[int] = []
        losses: List[float] = []
        hop_counts: Dict[int, int] = {}          # pid -> hops so far

        while self._heap:
            ev = heapq.heappop(self._heap)
            self.now = ev.time

            if ev.kind == "arrive":
                self._handle_arrive(
                    ev, agents, training, delivery_times, hops_list, hop_counts, losses
                )

        avg_delivery = float(np.mean(delivery_times)) if delivery_times else float("inf")
        avg_hops = float(np.mean(hops_list)) if hops_list else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0

        return {
            "avg_delivery_time": avg_delivery,
            "avg_hops": avg_hops,
            "delivered": len(delivery_times),
            "avg_loss": avg_loss,
        }

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_arrive(
        self,
        ev: _Event,
        agents: List[IRrAgent],
        training: bool,
        delivery_times: List[float],
        hops_list: List[int],
        hop_counts: Dict[int, int],
        losses: List[float],
    ) -> None:
        pkt: Packet = ev.data["pkt"]
        node: int = ev.data["node"]
        prev: Optional[int] = ev.data["prev"]

        # ---- decrement sender's queue immediately on arrival ----
        if prev is not None:
            agents[prev].queue_lengths[node] = max(
                0, agents[prev].queue_lengths[node] - 1
            )

        # ---- delivered? ----
        if node == pkt.destination:
            delivery_times.append(self.now - pkt.birth_time)
            hops_list.append(hop_counts.get(pkt.pid, 0))
            return

        agent = agents[node]

        # ---- drop packets caught in a routing loop ----
        if hop_counts.get(pkt.pid, 0) >= self.cfg.max_hops:
            return

        # ---- build state components BEFORE decision (stored in replay buffer) ----
        partial_state = agent.build_partial_state(pkt.destination)

        # ---- action: choose next hop (σ-greedy, paper Eq. 5) ----
        action_idx, next_node = agent.select_action(pkt.destination)

        # ---- cost: c_t = q + h  (paper §4.2) ----
        q = agent.queue_lengths[next_node]
        cost = q * self.cfg.queue_time_per_packet + self.cfg.transmission_time

        # ---- update sender's queue (enqueue) ----
        agent.queue_lengths[next_node] += 1

        # ---- schedule dequeue event (packet leaves sender after 'cost' ms) ----
        # The queue-length decrement is handled when packet arrives at next_node,
        # so no separate dequeue event is needed here.

        # ---- schedule arrival at next hop ----
        self._push(
            self.now + cost,
            "arrive",
            {"pkt": pkt, "node": next_node, "prev": node},
        )

        # ---- hop counter ----
        hop_counts[pkt.pid] = hop_counts.get(pkt.pid, 0) + 1

        # ---- increment tick ----
        agent.tick += 1

        # ---- training (Algorithm 1, lines 12-19) ----
        if training:
            # e_t = min_â Q̂_next(Ŝ, â)  (paper §4.3 / Eq. 6).
            # If next_node IS the destination no further routing happens there,
            # so the true remaining cost is 0.  The destination node's Q-values
            # are random (it never makes routing decisions and is never trained),
            # so using them would inflate targets for every last-hop transition.
            if next_node == pkt.destination:
                e_t = 0.0
            else:
                e_t = agents[next_node].min_q_value(pkt.destination)

            agent.store_transition(partial_state, cost, action_idx, e_t)

            # Gradient descent every L_c ticks
            if agent.tick % self.cfg.learning_cycle == 0:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

            # Periodic sub-GNN update (paper §3 step 5: "periodically sends FV")
            if agent.tick % self.cfg.gnn_update_period == 0:
                self._local_gnn_update(node, agents, n_iters=self.cfg.gnn_update_iters)

    # ------------------------------------------------------------------
    # GNN helpers
    # ------------------------------------------------------------------

    def _global_gnn_update(self, agents: List[IRrAgent], n_iters: int) -> None:
        """
        All nodes simultaneously exchange feature vectors for n_iters rounds.
        Used during initialisation (K iterations, paper §3).
        Also updates each agent's _nbr_fvs cache so train_step can use them.
        """
        for _ in range(n_iters):
            # Snapshot all FVs before any update so updates use t-1 values
            fvs = {n: agents[n].get_feature_vector() for n in range(self.topo.num_nodes)}
            for n in range(self.topo.num_nodes):
                nbr_fvs = {nb: fvs[nb] for nb in self.topo.adjacency[n]}
                fv_list = list(nbr_fvs.values())
                agents[n].sub_gnn.iterate(fv_list)
                agents[n]._nbr_fvs = [fv.detach() for fv in fv_list]

    def _local_gnn_update(
        self, node: int, agents: List[IRrAgent], n_iters: int
    ) -> None:
        """
        Node `node` exchanges FVs with its 1-hop neighbours (I_n iterations).
        This is the distributed periodic update described in paper §3 step 5.
        """
        nbr_fvs = {nb: agents[nb].get_feature_vector() for nb in self.topo.adjacency[node]}
        agents[node].gnn_iterate(nbr_fvs, n_iters)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _push(self, time: float, kind: str, data: dict) -> None:
        heapq.heappush(self._heap, _Event(time, kind, data))
