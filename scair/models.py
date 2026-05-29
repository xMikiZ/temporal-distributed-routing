"""
Neural network modules for ScaIR.

Two networks per agent (§4):
  SubGNN         -- distributed GNN with mean aggregation (§4.1)
  AttentionSubGNN -- same but uses softmax(dot-product) attention aggregation
  QNetwork       -- deep Q-network N_Q (§4.2)

Architecture details (paper §4.1 / §4.2):
  - Both f_w and g_w are feedforward NNs with ReLU activation.
  - Optimiser: RMSprop  (paper §4.3)
  - Activation: ReLU    (paper §4.3)

Aggregation variants:
  SubGNN:          agg = mean(V_neighbors)
  AttentionSubGNN: scores_i = dot(V_own, V_nbr_i)
                   weights  = softmax(scores)
                   agg      = weighted_sum(V_neighbors, weights)
  No learnable weight matrices in the attention — pure dot-product similarity.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


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


def make_shared_node_gnn(node_id: int, feature_length: int, template: "SubGNN") -> "SubGNN":
    """
    Create a per-node GNN that shares f_w / g_w weights with *template* but
    keeps an independent V state buffer.

    The returned object is an instance of a dynamically-created subclass of
    type(template) (SubGNN *or* AttentionSubGNN), so iterate() and
    get_output_trainable() use the correct aggregation method.  parameters()
    is overridden to return nothing — the template's optimizer covers f_w / g_w.

    Correct "shared weights" semantics
    -----------------------------------
    Without this, all agents share the same SubGNN object including V.
    Every agent's iterate() call overwrites V in sequence, so all nodes end
    up with the same feature vector (last writer wins).  With this factory,
    each node has its own V while the aggregation networks are truly shared.
    """
    base_cls = type(template)

    class _NodeGNN(base_cls):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.node_id = node_id
            self.feature_length = feature_length
            self.f_w = template.f_w          # shared tensor references
            self.g_w = template.g_w
            V_init = torch.zeros(feature_length)
            if node_id < feature_length:
                V_init[node_id] = 1.0
            self.register_buffer("V", V_init)

        def parameters(self, recurse: bool = True):
            return iter([])                  # weights owned by template; no double-count

    _NodeGNN.__name__ = f"Shared{base_cls.__name__}[{node_id}]"
    return _NodeGNN()


class AttentionSubGNN(SubGNN):
    """
    SubGNN variant that replaces mean aggregation with dot-product attention.

    Attention (no learnable weight matrices):
        scores_i = dot(V_own, V_nbr_i)          scalar per neighbour
        weights  = softmax(scores)               normalised
        agg      = sum_i( weights_i * V_nbr_i )  weighted sum

    Falls back to V_own zeros (no neighbours) or the single FV (one neighbour)
    exactly as SubGNN does.  f_w / g_w architecture and all other behaviour
    are inherited unchanged.
    """

    def _attend(self, V_own: torch.Tensor, nbr_Vs: List[torch.Tensor]) -> torch.Tensor:
        """Attention-weighted aggregation of neighbour feature vectors."""
        if not nbr_Vs:
            return torch.zeros(self.feature_length, device=V_own.device)
        if len(nbr_Vs) == 1:
            return nbr_Vs[0]
        nbr_stack = torch.stack(nbr_Vs)                    # [N, F_l]
        scores = (nbr_stack * V_own.unsqueeze(0)).sum(-1)  # [N]  dot products
        weights = F.softmax(scores, dim=0)                 # [N]
        return (weights.unsqueeze(1) * nbr_stack).sum(0)   # [F_l]

    def iterate(self, neighbour_Vs: List[torch.Tensor]) -> None:
        with torch.no_grad():
            agg = self._attend(self.V, [v.detach() for v in neighbour_Vs])
            inp = torch.cat([self.V.detach(), agg.detach()])
            self.V = self.f_w(inp).detach()

    def get_output_trainable(self, neighbour_Vs: List[torch.Tensor]) -> torch.Tensor:
        agg = self._attend(self.V.detach(), [v.detach() for v in neighbour_Vs])
        inp = torch.cat([self.V.detach(), agg.detach()])
        V_one_step = self.f_w(inp)
        return self.g_w(V_one_step)


def _make_init_v(node_id: int, feature_length: int,
                 init_v: Optional[torch.Tensor]) -> torch.Tensor:
    """Return V_init: either the supplied vector (zero-padded) or one-hot."""
    if init_v is not None:
        v = torch.zeros(feature_length)
        n = min(len(init_v), feature_length)
        v[:n] = init_v[:n].float()
        return v
    v = torch.zeros(feature_length)
    if node_id < feature_length:
        v[node_id] = 1.0
    return v


class PaperSubGNN(SubGNN):
    """
    Paper-faithful SubGNN: f_w is applied to each neighbour's V individually,
    then the outputs are averaged to form the new V_n.

    V_n^(t) = mean_y( f_w(V_y^(t-1)) )   -- f_w input dim = F_l

    Accepts optional init_v to override the default one-hot initialisation.
    """

    def __init__(
        self,
        node_id: int,
        num_nodes: int,
        feature_length: int,
        neural_units: int,
        init_v: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(node_id, num_nodes, feature_length, neural_units)
        self.f_w = nn.Sequential(
            nn.Linear(feature_length, neural_units),
            nn.ReLU(),
            nn.Linear(neural_units, feature_length),
        )
        self._V_init = _make_init_v(node_id, feature_length, init_v)
        self.register_buffer("V", self._V_init.clone())

    def iterate(self, neighbour_Vs: List[torch.Tensor]) -> None:
        with torch.no_grad():
            if neighbour_Vs:
                transformed = torch.stack([self.f_w(v.detach()) for v in neighbour_Vs])
                self.V = transformed.mean(dim=0).detach()
            else:
                self.V = torch.zeros(self.feature_length, device=self.V.device)

    def get_output_trainable(self, neighbour_Vs: List[torch.Tensor]) -> torch.Tensor:
        if neighbour_Vs:
            transformed = torch.stack([self.f_w(v.detach()) for v in neighbour_Vs])
            V_new = transformed.mean(dim=0)
        else:
            V_new = torch.zeros(self.feature_length, device=self.V.device)
        return self.g_w(V_new)

    def reset(self) -> None:
        self.V = self._V_init.clone().to(self.V.device)


class DotAttnSubGNN(PaperSubGNN):
    """
    GNN with scaled dot-product attention aggregation.

    score_y  = (V_n · f_w(V_y)) / sqrt(F_l)
    weights  = softmax(scores)
    V_n^new  = sum_y( weights_y * f_w(V_y) )

    No learnable attention parameters — attention derives from current V state.
    Accepts optional init_v for custom initialisation.
    """

    def iterate(self, neighbour_Vs: List[torch.Tensor]) -> None:
        with torch.no_grad():
            if not neighbour_Vs:
                self.V = torch.zeros(self.feature_length, device=self.V.device)
                return
            transformed = torch.stack([self.f_w(v.detach()) for v in neighbour_Vs])
            if len(neighbour_Vs) == 1:
                self.V = transformed[0].detach()
                return
            scores = (transformed * self.V.unsqueeze(0)).sum(-1) / math.sqrt(self.feature_length)
            weights = torch.softmax(scores, dim=0)
            self.V = (weights.unsqueeze(-1) * transformed).sum(0).detach()

    def get_output_trainable(self, neighbour_Vs: List[torch.Tensor]) -> torch.Tensor:
        if not neighbour_Vs:
            return self.g_w(torch.zeros(self.feature_length, device=self.V.device))
        t_for_scores = torch.stack([self.f_w(v.detach()) for v in neighbour_Vs]).detach()
        if len(neighbour_Vs) == 1:
            return self.g_w(self.f_w(neighbour_Vs[0].detach()))
        scores = (t_for_scores * self.V.detach().unsqueeze(0)).sum(-1) / math.sqrt(self.feature_length)
        weights = torch.softmax(scores, dim=0).detach()
        t_grad = torch.stack([self.f_w(v.detach()) for v in neighbour_Vs])
        V_new = (weights.unsqueeze(-1) * t_grad).sum(0)
        return self.g_w(V_new)


class LearnableAttnSubGNN(PaperSubGNN):
    """
    GNN with learnable query-projection attention.

    q        = W_q * V_n                        (learnable W_q)
    score_y  = (q · f_w(V_y)) / sqrt(F_l)
    weights  = softmax(scores)
    V_n^new  = sum_y( weights_y * f_w(V_y) )

    Accepts optional init_v for custom initialisation.
    """

    def __init__(
        self,
        node_id: int,
        num_nodes: int,
        feature_length: int,
        neural_units: int,
        init_v: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(node_id, num_nodes, feature_length, neural_units, init_v=init_v)
        self.W_q = nn.Linear(feature_length, feature_length, bias=False)

    def _attention_weights(self, transformed: torch.Tensor,
                           v_query: torch.Tensor) -> torch.Tensor:
        q = self.W_q(v_query)
        scores = (transformed * q.unsqueeze(0)).sum(-1) / math.sqrt(self.feature_length)
        return torch.softmax(scores, dim=0)

    def iterate(self, neighbour_Vs: List[torch.Tensor]) -> None:
        with torch.no_grad():
            if not neighbour_Vs:
                self.V = torch.zeros(self.feature_length, device=self.V.device)
                return
            transformed = torch.stack([self.f_w(v.detach()) for v in neighbour_Vs])
            if len(neighbour_Vs) == 1:
                self.V = transformed[0].detach()
                return
            weights = self._attention_weights(transformed, self.V)
            self.V = (weights.unsqueeze(-1) * transformed).sum(0).detach()

    def get_output_trainable(self, neighbour_Vs: List[torch.Tensor]) -> torch.Tensor:
        if not neighbour_Vs:
            return self.g_w(torch.zeros(self.feature_length, device=self.V.device))
        if len(neighbour_Vs) == 1:
            return self.g_w(self.f_w(neighbour_Vs[0].detach()))
        # Score with detached f_w outputs; W_q retains grad
        t_for_scores = torch.stack([self.f_w(v.detach()) for v in neighbour_Vs]).detach()
        q = self.W_q(self.V.detach())           # grad flows through W_q
        scores = (t_for_scores * q.unsqueeze(0)).sum(-1) / math.sqrt(self.feature_length)
        weights = torch.softmax(scores, dim=0)  # grad through W_q
        t_grad = torch.stack([self.f_w(v.detach()) for v in neighbour_Vs])
        V_new = (weights.unsqueeze(-1) * t_grad).sum(0)
        return self.g_w(V_new)


class FixedSubGNN(nn.Module):
    """SubGNN replacement with a constant, non-learned feature vector.

    Replaces the GNN output V_n with a hand-crafted topology encoding so we
    can ablate whether the GNN is actually useful.  All SubGNN interface
    methods are preserved but iterate() and reset() are no-ops.
    """

    def __init__(self, vector: torch.Tensor) -> None:
        super().__init__()
        self.feature_length = len(vector)
        self.f_w = None
        self.g_w = None
        self.register_buffer("V", vector.float())

    def iterate(self, neighbour_Vs) -> None:
        pass

    def get_output(self) -> torch.Tensor:
        return self.V

    def get_output_trainable(self, neighbour_Vs) -> torch.Tensor:
        return self.V  # fixed buffer — no grad path through it

    def reset(self) -> None:
        pass

    def parameters(self, recurse: bool = True):
        return iter([])  # nothing to train


class OneHotSubGNN(FixedSubGNN):
    """V_n = one-hot at position node_id, zero-padded to feature_length.

    Gives the Q-network a unique but topology-free node identity.
    Constructor accepts *neighbors* for API symmetry with NeighborMaskSubGNN.
    """

    def __init__(self, node_id: int, neighbors, feature_length: int) -> None:
        v = torch.zeros(feature_length)
        if node_id < feature_length:
            v[node_id] = 1.0
        super().__init__(v)


class NeighborMaskSubGNN(FixedSubGNN):
    """V_n = binary mask with 1 at each neighbour's index, 0 elsewhere.

    Encodes immediate topology (degree, neighbour identities) without learning.
    """

    def __init__(self, node_id: int, neighbors, feature_length: int) -> None:
        v = torch.zeros(feature_length)
        for nb in neighbors:
            if nb < feature_length:
                v[nb] = 1.0
        super().__init__(v)


def make_fixed_gnn(init_vs: Dict[int, torch.Tensor]):
    """Factory: returns a FixedSubGNN class whose V is taken from init_vs[node_id].

    The returned class has the same constructor signature as OneHotSubGNN /
    NeighborMaskSubGNN, so it works with build_agents_no_gnn.
    """
    class _TopoFixed(FixedSubGNN):
        def __init__(self, node_id: int, neighbors, feature_length: int) -> None:
            iv = init_vs[node_id]
            v = torch.zeros(feature_length)
            n = min(len(iv), feature_length)
            v[:n] = iv[:n].float()
            super().__init__(v)
    return _TopoFixed


class QNetwork(nn.Module):
    """
    Q-Network (N_Q) for a single agent.

    Input (paper §4.2, all concatenated):
        1. Destination one-hot          [max_nodes]
        2. Queue lengths (padded)       [max_degree]
        3. Feature vector from SubGNN   [feature_length]
        4. Action history               [action_history_len * max_degree]

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
    ) -> None:
        super().__init__()
        self.max_degree = max_degree

        input_dim = (
            max_nodes                           # destination one-hot
            + max_degree                        # queue lengths
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
