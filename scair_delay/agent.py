"""
IRr (Intelligent Router) agent for ScaIR.

Each agent n owns:
  - SubGNN  (N_Gsub) -- feature extraction
  - QNetwork (N_Q)   -- routing decisions
  - ReplayMemory D_n
  - Per-interface queue-length counters

The combined parameter set of SubGNN + QNetwork is optimised jointly
with a single RMSprop optimiser (paper §4.3, Eq. 7-8):
    theta*_n, g*_n <- GradientDescent( (y - Q_n(s_n, a_n | theta_n, g_n))^2 )
"""

import math
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ScaIRConfig
from .models import QNetwork, SubGNN


# ---------------------------------------------------------------------------
# Replay memory
# ---------------------------------------------------------------------------

class Transition:
    __slots__ = ("partial_state", "cost", "action", "estimated_time")

    def __init__(
        self,
        partial_state: torch.Tensor,   # dest_oh + queue_lengths + action_history
        cost: float,
        action: int,
        estimated_time: float,
    ) -> None:
        self.partial_state = partial_state
        self.cost = cost
        self.action = action
        self.estimated_time = estimated_time


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self.buffer.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class IRrAgent:
    """
    One intelligent router node.

    State space (paper §4.2):  S_n = {d_p, O_n, K_n}
        d_p  -- destination one-hot                    [max_nodes]
        O_n  -- queue lengths per interface (padded)   [max_degree]
               + action history k one-hots (padded)    [k * max_degree]
        K_n  -- o_n from SubGNN                        [feature_length]

    Action space:  A_n = W_n  (set of neighbour nodes)

    Cost:  c_t = q + h  (queuing time + transmission delay, paper §4.2)

    Target (Eq. 6):  y = c + gamma * min_â Q̂(Ŝ, â | θ̂)
        where Q̂ is the *next* agent's Q-network evaluated on the state
        it will be in when it receives the packet.
    """

    def __init__(
        self,
        node_id: int,
        neighbours: List[int],
        num_nodes: int,
        cfg: ScaIRConfig,
        link_delays: Optional[Dict[int, float]] = None,
        max_delay: float = 1.0,
    ) -> None:
        self.node_id = node_id
        self.neighbours = neighbours          # ordered list of neighbour node IDs
        self.degree = len(neighbours)
        self._nbr_to_idx: Dict[int, int] = {n: i for i, n in enumerate(neighbours)}
        self.cfg = cfg

        # Per-link propagation delays [max_degree], normalised to [0, 1] by max_delay.
        # Only populated (and used) when cfg.delay_input is True.
        self._delay_vec: torch.Tensor = torch.zeros(cfg.max_degree)
        if cfg.delay_input and link_delays:
            for i, nbr in enumerate(neighbours):
                if i < cfg.max_degree:
                    self._delay_vec[i] = float(link_delays.get(nbr, 0.0)) / max(max_delay, 1e-6)

        # Neural networks
        self.sub_gnn = SubGNN(node_id, num_nodes, cfg.feature_length, cfg.neural_units)
        self.q_net = QNetwork(
            max_nodes=cfg.max_nodes,
            max_degree=cfg.max_degree,
            feature_length=cfg.feature_length,
            neural_units=cfg.neural_units,
            action_history_len=cfg.action_history_len,
            delay_input=cfg.delay_input,
        )
        # Target network: frozen copy of q_net used to compute e_t.
        # Updated periodically via update_target(); never trained directly.
        self.q_net_target = QNetwork(
            max_nodes=cfg.max_nodes,
            max_degree=cfg.max_degree,
            feature_length=cfg.feature_length,
            neural_units=cfg.neural_units,
            action_history_len=cfg.action_history_len,
            delay_input=cfg.delay_input,
        )
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        for p in self.q_net_target.parameters():
            p.requires_grad_(False)

        # Joint optimiser for SubGNN + QNetwork (paper Eq. 8, RMSprop).
        # GNN uses a lower LR than Q-net: sharing Q-net's LR=0.1 would shift
        # the GNN output too fast and destabilise the Q-net input.
        self.optimizer = torch.optim.RMSprop([
            {"params": self.q_net.parameters(),   "lr": cfg.learning_rate_initial},
            {"params": self.sub_gnn.parameters(), "lr": cfg.gnn_learning_rate},
        ])

        # Replay memory
        self.memory = ReplayMemory(cfg.memory_size)

        # Per-interface queue lengths (updated by environment)
        self.queue_lengths: Dict[int, int] = {n: 0 for n in neighbours}

        # Action history: circular buffer of the last k action indices
        self._action_history: deque = deque(
            [-1] * cfg.action_history_len, maxlen=cfg.action_history_len
        )

        # Most-recent neighbour feature vectors — set by gnn_iterate / environment
        # and used in train_step so f_w receives gradient updates (paper Eq. 8).
        self._nbr_fvs: List[torch.Tensor] = []

        # Decision tick counter (used for learning_cycle and GNN update triggers)
        self.tick: int = 0

        # Exploration rate (epsilon-greedy)
        self.sigma: float = cfg.sigma_initial

        # UCB action counts: dest -> [count_per_action]
        # Accumulated across episodes (not reset) so UCB estimates improve over time.
        self._ucb_counts: Dict[int, List[int]] = {}

        self._num_nodes = num_nodes

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def build_partial_state(self, destination: int) -> torch.Tensor:
        """
        Build the GNN-free part of the state: {d_p, O_n, action_history}.
        This is what gets stored in the replay buffer so the buffer is
        decoupled from the GNN weights (which change over training).
        """
        # 1. Destination one-hot  [max_nodes]
        dest_oh = torch.zeros(self.cfg.max_nodes)
        if destination < self.cfg.max_nodes:
            dest_oh[destination] = 1.0

        # 2. Queue lengths, zero-padded to max_degree  [max_degree]
        q_vec = torch.zeros(self.cfg.max_degree)
        for i, nbr in enumerate(self.neighbours):
            if i < self.cfg.max_degree:
                q_vec[i] = float(self.queue_lengths[nbr])

        # 3. Action history: k one-hot vectors of size max_degree
        hist = torch.zeros(self.cfg.action_history_len * self.cfg.max_degree)
        for step, act in enumerate(self._action_history):
            if act >= 0 and act < self.cfg.max_degree:
                hist[step * self.cfg.max_degree + act] = 1.0

        if self.cfg.delay_input:
            return torch.cat([dest_oh, q_vec, self._delay_vec, hist])
        return torch.cat([dest_oh, q_vec, hist])

    def build_state(self, destination: int) -> torch.Tensor:
        """
        Full state S_n = {d_p, O_n, K_n, action_history} for action selection.
        The GNN output (K_n) is detached here — no gradient needed for inference.
        """
        partial = self.build_partial_state(destination)
        with torch.no_grad():
            fv = self.sub_gnn.get_output().detach()
        # Insert fv after queue lengths (and delays if enabled), before action history.
        split = self.cfg.max_nodes + self.cfg.max_degree
        if self.cfg.delay_input:
            split += self.cfg.max_degree
        return torch.cat([partial[:split], fv, partial[split:]])

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, destination: int) -> Tuple[int, int]:
        """
        Dispatches to epsilon-greedy or UCB depending on cfg.action_method.
        Returns (action_index, neighbour_node_id).
        """
        if self.cfg.action_method == "ucb":
            idx = self._select_action_ucb(destination)
        else:
            idx = self._select_action_epsilon_greedy(destination)

        self._action_history.append(idx)
        return idx, self.neighbours[idx]

    def _select_action_epsilon_greedy(self, destination: int) -> int:
        """sigma-greedy policy (paper Eq. 5)."""
        if random.random() < self.sigma:
            return random.randrange(self.degree)
        state = self.build_state(destination)
        with torch.no_grad():
            q_vals = self.q_net(state.unsqueeze(0))[0, : self.degree]
        return int(q_vals.argmin().item())

    def _select_action_ucb(self, destination: int) -> int:
        """
        Lower Confidence Bound (LCB) for cost minimisation:
            a* = argmin_a [ Q(s,a) - c * sqrt( ln(N+1) / n_a ) ]

        Actions with lower estimated cost OR fewer visits are preferred.
        Unvisited actions are tried first (count=0 → infinite bonus).
        Counts accumulate across episodes.
        """
        if destination not in self._ucb_counts:
            self._ucb_counts[destination] = [0] * self.degree

        counts = self._ucb_counts[destination]

        # Force exploration of any unvisited action first
        for idx, c in enumerate(counts):
            if c == 0:
                counts[idx] += 1
                return idx

        state = self.build_state(destination)
        with torch.no_grad():
            q_vals = self.q_net(state.unsqueeze(0))[0, : self.degree].cpu().numpy()

        total = sum(counts)
        bonus = self.cfg.ucb_c * np.sqrt(math.log(total + 1) / np.array(counts, dtype=float))
        lcb = q_vals - bonus
        idx = int(np.argmin(lcb))
        counts[idx] += 1
        return idx

    # ------------------------------------------------------------------
    # Min-Q for next-hop feedback  (e_t in the paper)
    # ------------------------------------------------------------------

    def min_q_value(self, destination: int) -> float:
        """
        Compute min_â Q̂_n(s_n, â) using the TARGET network for the given destination.
        Used by the *previous* node to compute the stable target y = c + gamma * e_t.
        Using the target network (not the live q_net) prevents targets from shifting
        every gradient step, stabilising training.
        """
        state = self.build_state(destination)
        with torch.no_grad():
            q_vals = self.q_net_target(state.unsqueeze(0))[0, : self.degree]
        return float(q_vals.min().item())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def store_transition(
        self,
        partial_state: torch.Tensor,
        cost: float,
        action_idx: int,
        estimated_time: float,
    ) -> None:
        self.memory.push(Transition(partial_state.detach(), cost, action_idx, estimated_time))

    def train_step(self) -> Optional[float]:
        """
        Sample a random mini-batch and perform one gradient-descent step.

        Loss (paper Eq. 7):
            L_t = ( y - Q_n(s_n, a_n | theta_n, g_n) )^2
        Target (paper Eq. 6):
            y = c + gamma * min_â Q̂(Ŝ, â | θ̂)
        where e_t = min_â Q̂(.) was already computed by the next agent and
        stored in the transition.

        The GNN output (fv = g_w(V)) is recomputed here WITHOUT detaching so
        that gradients flow back to g_w (and, via the iterate chain, to f_w).
        This implements the joint θ_n, g_n update from paper Eq. 8.
        """
        if len(self.memory) < self.cfg.batch_size:
            return None

        batch = self.memory.sample(self.cfg.batch_size)
        B = len(batch)

        partial_states = torch.stack([t.partial_state for t in batch])   # [B, partial_dim]
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        costs = torch.tensor([t.cost for t in batch], dtype=torch.float32)
        e_ts = torch.tensor([t.estimated_time for t in batch], dtype=torch.float32)

        # Live differentiable fv: gradients flow to f_w and g_w (paper Eq. 8 joint update).
        # Using the current GNN state (one fv for the whole batch) is consistent with
        # having a single SubGNN per agent; the same approach is used during action selection.
        fv_live = self.sub_gnn.get_output_trainable(self._nbr_fvs)        # [F_l]
        fv_expanded = fv_live.unsqueeze(0).expand(B, -1)                  # [B, F_l]

        split = self.cfg.max_nodes + self.cfg.max_degree
        if self.cfg.delay_input:
            split += self.cfg.max_degree
        states = torch.cat(
            [partial_states[:, :split], fv_expanded, partial_states[:, split:]], dim=1
        )   # [B, input_dim]

        # Target y = c + gamma * e_t  (Eq. 6)
        y = costs + self.cfg.discount_factor * e_ts

        # Q-value of the action actually taken
        q_all = self.q_net(states)                                            # [B, max_degree]
        q_taken = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)            # [B]

        loss = F.mse_loss(q_taken, y.detach())

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping: stabilises training with the high initial LR=0.1
        torch.nn.utils.clip_grad_norm_(
            list(self.sub_gnn.parameters()) + list(self.q_net.parameters()), max_norm=1.0
        )
        self.optimizer.step()

        return float(loss.item())

    def update_target(self) -> None:
        """Hard-copy live q_net weights into q_net_target."""
        self.q_net_target.load_state_dict(self.q_net.state_dict())

    def set_learning_rate(self, lr: float) -> None:
        # Only update the Q-net group (index 0); GNN LR stays fixed.
        self.optimizer.param_groups[0]["lr"] = lr

    def decay_sigma(self) -> None:
        self.sigma = max(self.cfg.sigma_min, self.sigma - self.cfg.sigma_decrement)

    # ------------------------------------------------------------------
    # Sub-GNN interface
    # ------------------------------------------------------------------

    def get_feature_vector(self) -> torch.Tensor:
        """Return V_n (detached copy) for sharing with neighbours."""
        return self.sub_gnn.V.detach().clone()

    def gnn_iterate(self, neighbour_fvs: Dict[int, torch.Tensor], n_iters: int) -> None:
        """
        Run n_iters message-passing steps using the given neighbour FVs.
        Neighbour FVs are treated as fixed (collected before this call).
        Also caches the FVs so train_step can pass them to get_output_trainable.
        """
        fv_list = [neighbour_fvs[nbr] for nbr in self.neighbours if nbr in neighbour_fvs]
        self._nbr_fvs = [fv.detach() for fv in fv_list]
        for _ in range(n_iters):
            self.sub_gnn.iterate(fv_list)

    def reset_episode(self) -> None:
        """Reset per-episode state: queues, action history, GNN buffer.

        tick is intentionally NOT reset here.  Algorithm 1 uses a single
        global decision counter t that runs across episodes; resetting it
        per episode means low-traffic agents (fewer than learning_cycle
        decisions per episode) never reach the modulo condition and
        therefore never call train_step or the GNN periodic update.
        """
        self.queue_lengths = {n: 0 for n in self.neighbours}
        self._action_history = deque(
            [-1] * self.cfg.action_history_len, maxlen=self.cfg.action_history_len
        )
        self._nbr_fvs = []
        self.sub_gnn.reset()
