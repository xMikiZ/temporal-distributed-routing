"""
ScaIR hyperparameters.

All values match Table 1 of the paper unless noted.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScaIRConfig:
    # ----- Sub-GNN (N_Gsub) -----
    feature_length: int = 128       # F_l: feature vector length
    neural_units: int = 64          # N_u: hidden units per layer in f_w and g_w
    gnn_init_iters: int = 8         # K: GNN iterations during initialisation
    gnn_update_iters: int = 3       # I_n: GNN iterations during periodic update

    # ----- DRL (N_Q) -----
    discount_factor: float = 1.0    # D_f / gamma
    # Paper Table 1: L_r = 0.1 for ep 1-10, then 0.001.
    # LR=0.1 kills all ReLU neurons by episode 1 in practice (Q-values are in
    # ms scale so random targets can be tens of ms off, causing huge overshoot).
    # Using 0.001 throughout avoids dead neurons without a two-stage schedule.
    learning_rate_initial: float = 0.001
    learning_rate: float = 0.001         # L_r after episode 10

    # GNN learning rate (separate from Q-net LR).
    # The paper trains both jointly but doesn't give separate rates.
    # Sharing LR=0.1 with the Q-net causes the GNN output to shift too
    # fast (destabilising the Q-net input), so a lower rate is needed.
    gnn_learning_rate: float = 0.0001
    sigma_initial: float = 0.9      # sigma-greedy: initial random-action probability
    sigma_min: float = 0.1
    sigma_decrement: float = 0.05   # sigma decreases by this every sigma_decay_freq episodes
    sigma_decay_freq: int = 10      # episodes between sigma decrements

    # ----- Replay & training -----
    memory_size: int = 200          # M_s
    batch_size: int = 64            # B_s
    learning_cycle: int = 10        # L_c: ticks between weight updates
    tau: float = 0.01               # Polyak coefficient for soft target updates

    # ----- Episode / traffic -----
    max_episodes: int = 400
    packets_per_episode: int = 50   # P_n
    generation_interval: float = 0.5  # G_i: mean Poisson inter-arrival time (ms)
    # D_r: hot-spot fraction (node 0 -> last node).
    # Paper Fig.9 shows ScaIR beats OSPF only when D_r >= 0.4.
    # With D_r=0.0 there is no congestion and OSPF (shortest path) is optimal.
    distribution_ratio: float = 0.5

    # Maximum hops before a packet is dropped (loop prevention).
    # With sigma=0.9 at start, random actions produce loops; without a limit
    # those packets inflate avg_delivery_time indefinitely.
    max_hops: int = 50

    # ----- Simulation -----
    transmission_time: float = 1.0       # ms per hop (fixed in paper)
    queue_time_per_packet: float = 1.0   # additional ms per packet already in queue

    # ----- Encoding -----
    # Fixed upper-bound dimensions so all agents share the same N_Q input shape.
    # "The length of the one-hot vector is usually fixed at an upper limit" (paper §4.2).
    max_nodes: int = 30       # upper bound on topology size for one-hot destination
    max_degree: int = 10      # upper bound on node degree (for padding)
    action_history_len: int = 5   # k: number of past actions kept in state

    # ----- Q-network input options -----
    # When True, adds per-link propagation delays [max_degree] to Q-net input.
    delay_input: bool = False

    # ----- GNN init options -----
    # When True, V_0 is seeded with normalised neighbour delays instead of pure one-hot.
    delay_init: bool = False

    # ----- Action selection -----
    # "epsilon_greedy" (default) or "ucb"
    action_method: str = "epsilon_greedy"
    # UCB exploration constant c (Lower Confidence Bound for cost minimisation).
    # The selected action is argmin_a [ Q(s,a) - c * sqrt(ln(N+1) / n_a) ].
    # Higher c -> more exploration.  Needs tuning relative to Q-value scale (ms).
    ucb_c: float = 2.0

    # ----- Misc -----
    seed: Optional[int] = None
    save_dir: str = "checkpoints"
    log_interval: int = 10    # print every N episodes
    gnn_update_period: int = 10  # decision ticks between GNN periodic updates
