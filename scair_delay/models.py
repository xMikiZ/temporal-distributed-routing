"""
Neural network modules for ScaIR.

Two networks per agent (§4):
  SubGNN  -- distributed graph neural network N_Gsub (§4.1)
  QNetwork -- deep Q-network N_Q                     (§4.2)

Architecture details (paper §4.1 / §4.2):
  - Both f_w and g_w are feedforward NNs with ReLU activation.
  - "Same architecture but different parameters for different nodes."
  - Optimiser: RMSprop  (paper §4.3: "root mean square propagation")
  - Activation: ReLU    (paper §4.3: "rectified linear unit")

Design decision (not specified in paper):
  f_w^n takes concat(V_own, mean(V_neighbors)) as input.
  Mean aggregation is standard GCN practice and handles variable-degree nodes.
"""

import torch
import torch.nn as nn
from typing import List


class SubGNN(nn.Module):
    """
    Sub-GNN (N_Gsub) for a single node.

    Message-passing equations (paper Eq. 2-4):
        V_n^(t=0)  = init(i_n)                          -- one-hot node ID, zero-padded
        V_n^(t)    = f_w^n({V_y^(t-1) | y in N_n})     -- update from neighbours
        o_n        = g_w^n(V_n^(t->K))                  -- output feature vector

    Here f_w^n(.) is implemented as:
        input  = concat(V_n, mean(V_neighbors))   [2 * F_l]
        output = new hidden representation         [F_l]
    """

    def __init__(
        self,
        node_id: int,
        num_nodes: int,
        feature_length: int,
        neural_units: int,
    ) -> None:
        super().__init__()
        self.node_id = node_id
        self.feature_length = feature_length

        # f_w: concat(V_own [F_l], mean_neighbours [F_l]) -> V_new [F_l]
        self.f_w = nn.Sequential(
            nn.Linear(2 * feature_length, neural_units),
            nn.ReLU(),
            nn.Linear(neural_units, feature_length),
        )

        # g_w: V [F_l] -> o_n [F_l]
        self.g_w = nn.Sequential(
            nn.Linear(feature_length, neural_units),
            nn.ReLU(),
            nn.Linear(neural_units, feature_length),
        )

        # Initialise V as one-hot of node_id, zero-padded to feature_length (Eq. 4)
        V_init = torch.zeros(feature_length)
        if node_id < feature_length:
            V_init[node_id] = 1.0
        # V is a plain tensor updated with no_grad; it is NOT a parameter
        self.register_buffer("V", V_init)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def iterate(self, neighbour_Vs: List[torch.Tensor]) -> None:
        """
        One message-passing step.  Updates self.V in-place.

        Args:
            neighbour_Vs: feature vectors V_y for each y in N_n.
        """
        if neighbour_Vs:
            agg = torch.stack(neighbour_Vs).mean(dim=0)
        else:
            agg = torch.zeros(self.feature_length, device=self.V.device)

        inp = torch.cat([self.V.detach(), agg.detach()])   # [2 * F_l]
        # Keep f_w update detached so V stays numerically stable across the
        # K=8 init iterations and gnn_update_period periodic updates.
        # g_w IS trained via backprop (see train_step in agent.py): even with
        # V.requires_grad=False, g_w.weight receives gradients through fv=g_w(V).
        with torch.no_grad():
            self.V = self.f_w(inp).detach()

    def get_output(self) -> torch.Tensor:
        """Return o_n = g_w(V).  Differentiable (used in backward pass)."""
        return self.g_w(self.V)

    def get_output_trainable(self, neighbour_Vs: List[torch.Tensor]) -> torch.Tensor:
        """
        Like get_output but runs one differentiable f_w step before g_w,
        so gradients flow through f_w during train_step (paper Eq. 8: g_n updated jointly).

        V and neighbour aggregation are detached — only f_w and g_w weights receive grads.
        """
        if neighbour_Vs:
            agg = torch.stack(neighbour_Vs).mean(dim=0)
        else:
            agg = torch.zeros(self.feature_length, device=self.V.device)
        inp = torch.cat([self.V.detach(), agg.detach()])
        V_one_step = self.f_w(inp)          # grad flows to f_w.weight
        return self.g_w(V_one_step)         # grad flows to g_w.weight too

    def reset(self) -> None:
        """Reset V to the initial one-hot representation (start of episode)."""
        V_init = torch.zeros(self.feature_length, device=self.V.device)
        if self.node_id < self.feature_length:
            V_init[self.node_id] = 1.0
        self.V = V_init.detach()


class QNetwork(nn.Module):
    """
    Q-Network (N_Q) for a single agent.

    Input (paper §4.2, all concatenated):
        1. Destination one-hot          [max_nodes]
        2. Queue lengths (padded)       [max_degree]
        3. Per-link delays (padded)     [max_degree]  -- only when delay_input=True
        4. Feature vector from SubGNN   [feature_length]
        5. Action history               [action_history_len * max_degree]

    Output:
        Q-values for each possible action (neighbour), padded to [max_degree].
        Only the first `degree` outputs correspond to valid neighbours.

    "The sizes of the output layer and the agent's action space |A_n| are identical."
    We use max_degree and mask invalid outputs at decision time.
    """

    def __init__(
        self,
        max_nodes: int,
        max_degree: int,
        feature_length: int,
        neural_units: int,
        action_history_len: int,
        delay_input: bool = False,
    ) -> None:
        super().__init__()
        self.max_degree = max_degree

        input_dim = (
            max_nodes                           # destination one-hot
            + max_degree                        # queue lengths
            + (max_degree if delay_input else 0)  # per-link propagation delays
            + feature_length                    # GNN feature vector
            + action_history_len * max_degree   # action history one-hots
        )

        self.net = nn.Sequential(
            nn.Linear(input_dim, neural_units),
            nn.ReLU(),
            nn.Linear(neural_units, neural_units),
            nn.ReLU(),
            nn.Linear(neural_units, max_degree),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, input_dim] or [input_dim].  Returns Q-values [batch, max_degree]."""
        return self.net(x)
