# ScaIR: Scalable Intelligent Routing based on Distributed Graph Reinforcement Learning

A Python implementation of the ScaIR algorithm from:

> **ScaIR: Scalable Intelligent Routing based on Distributed Graph Reinforcement Learning**
> Jing Zhang, Jianfeng Guan, Kexian Liu, Yizhong Hu, Ao Shen, Yuyin Ma
> *Computer Networks 257 (2025) 110915*
> https://doi.org/10.1016/j.comnet.2024.110915

ScaIR is a fully distributed multi-agent routing algorithm. Each router acts as an independent agent that combines a local sub-GNN (for global network awareness via feature-vector exchange with one-hop neighbours) and a DQN (for routing decisions), with the goal of minimising average packet delivery time.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Data Format](#data-format)
- [Configuration Parameters](#configuration-parameters)
- [Implementation Notes and Deviations from the Paper](#implementation-notes-and-deviations-from-the-paper)

---

## Overview

The paper proposes a distributed routing algorithm where each router:

1. Maintains a **sub-GNN** (N_Gsub) that iteratively exchanges compressed feature vectors with one-hop neighbours, producing a representation of the global network state.
2. Feeds this feature vector, along with local queue lengths, the packet destination, and an action history, into a **Q-network** (N_Q) that selects the next-hop neighbour.
3. Trains both networks jointly via experience replay, minimising the mean-squared error between the predicted delivery time and the observed cost plus the next agent's estimated future cost.

This implementation replaces the OMNet++ C++ simulation platform used in the paper with a self-contained Python event-driven simulator, making the code easy to run without external network simulation software.

Currently supported topologies:
- **Abilene** (11 nodes, 14 bidirectional links) — Internet2 backbone, real TMs
- **GEANT** (23 nodes, 37 bidirectional links) — European research network, real TMs
- **BRAIN** (9 nodes, aggregated Berlin backbone) — SNDlib dataset, real 1-min TMs
- **Germany50** (50 nodes, 88 bidirectional links) — DFN backbone, real 5-min TMs

---

## Project Structure

```
temporal-distributed-routing/
├── train.py              # Training script
├── evaluate.py           # Evaluation against baselines + training curve plotting
├── requirements.txt
├── scair/
│   ├── config.py         # ScaIRConfig dataclass (all hyperparameters)
│   ├── agent.py          # IRrAgent: sub-GNN + Q-network per router
│   ├── environment.py    # Event-driven packet routing simulator
│   ├── models.py         # SubGNN, PaperSubGNN, AttentionSubGNN, QNetwork, etc.
│   └── data_loader.py    # Topology and traffic matrix loaders
├── experiments/
│   ├── comparison_experiment.py      # Exp 1/3/4 — D_r sweep (reusable across topos)
│   ├── topology_robustness_variants.py # Exp 2 — topology mutation robustness
│   ├── ablation_no_gnn.py            # Exp 5 — no-GNN fixed-encoding ablation
│   ├── transfer_experiment.py        # Exp 6 — transfer learning across topologies
│   └── paper_vs_ours_experiment.py   # Exp 7 — paper f_w vs our f_w formulation
└── data/
    ├── ABI/              # Abilene (11 nodes)
    ├── GEA/              # GEANT (23 nodes)
    ├── BRA/              # BRAIN Berlin backbone (9 nodes, aggregated)
    └── GER50/            # Germany50 DFN backbone (50 nodes)
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- `torch >= 2.0.0`
- `numpy >= 1.24.0`
- `matplotlib` (optional, for plotting training curves)

---

## Usage

### Training

```bash
python train.py --topo data/ABI/Topology.txt \
                --tm_dir data/ABI/TrafficMatrix \
                --link_weights data/ABI/link_weight.json \
                --episodes 400 \
                --packets 50 \
                --feature_length 128 \
                --save_dir checkpoints/ABI
```

To resume from a checkpoint:

```bash
python train.py --topo data/ABI/Topology.txt \
                --tm_dir data/ABI/TrafficMatrix \
                --resume checkpoints/ABI/episode_0200.pt \
                --save_dir checkpoints/ABI
```

The episode number is inferred automatically from the checkpoint filename (e.g. `episode_0200.pt` → resume from episode 201). The `history.json` file in the save directory is appended with new episodes so the full training curve is preserved across resumed runs.

**GEANT example:**

```bash
python train.py --topo data/GEA/Topology.txt \
                --tm_dir data/GEA/TrafficMatrix \
                --link_weights data/GEA/link_weight.json \
                --feature_length 128 \
                --neural_units 64 \
                --gnn_iters 3 \
                --save_dir checkpoints/GEA
```

#### Training arguments

| Argument | Default | Description |
|---|---|---|
| `--topo` | required | Path to topology file |
| `--tm_dir` | required | Directory containing traffic matrix files |
| `--link_weights` | — | Path to `link_weight.json` for per-link delay overrides |
| `--episodes` | 400 | Total training episodes |
| `--packets` | 50 | Packets per episode (P_n) |
| `--feature_length` | 128 | GNN feature vector length (F_l) |
| `--neural_units` | 64 | Hidden units in f_w, g_w, and N_Q (N_u) |
| `--gnn_iters` | 3 | GNN update iterations per tick (I_n) |
| `--gen_interval` | 0.5 | Poisson mean inter-arrival time in ms (G_i) |
| `--dist_ratio` | 0.5 | Fraction of traffic forced to the hot-spot pair (D_r) |
| `--save_dir` | `checkpoints` | Directory for checkpoints and history |
| `--save_freq` | 50 | Save a checkpoint every N episodes (0 = end only) |
| `--resume` | — | Checkpoint file to resume from |
| `--seed` | — | Random seed for reproducibility |
| `--log_interval` | 10 | Print stats every N episodes |

#### Checkpoints

Each checkpoint is a `.pt` file containing, for every agent node, its sub-GNN weights, Q-network weights, and current σ value:

```python
{
  "0": {"sub_gnn": state_dict, "q_net": state_dict, "sigma": float},
  "1": {...},
  ...
}
```

Training statistics (average delivery time, hops, loss per episode) are saved to `history.json` in the save directory.

---

### Evaluation

Evaluate a trained checkpoint against OSPF and Q-routing baselines:

```bash
python evaluate.py --topo data/ABI/Topology.txt \
                   --tm_dir data/ABI/TrafficMatrix \
                   --checkpoint checkpoints/ABI/episode_0400.pt \
                   --episodes 100
```

Plot the training curve from a saved history file:

```bash
python evaluate.py --topo data/ABI/Topology.txt \
                   --tm_dir data/ABI/TrafficMatrix \
                   --history checkpoints/ABI/history.json \
                   --skip_eval
```

#### Evaluation arguments

| Argument | Default | Description |
|---|---|---|
| `--topo` | required | Path to topology file |
| `--tm_dir` | required | Directory containing traffic matrix files |
| `--link_weights` | — | Path to `link_weight.json` |
| `--checkpoint` | — | Trained ScaIR checkpoint to evaluate |
| `--history` | — | `history.json` file to plot |
| `--save_plot` | — | Save the training plot to this file path |
| `--skip_eval` | — | Skip evaluation, only plot history |
| `--episodes` | 100 | Evaluation episodes |
| `--packets` | 50 | Packets per episode |
| `--feature_length` | 128 | Must match the checkpoint |
| `--neural_units` | 64 | Must match the checkpoint |
| `--seed` | 42 | Random seed |

#### Baselines

- **OSPF:** Dijkstra shortest-path by hop count, computed offline from the full global topology. Routes packets through the same event-driven simulator so queuing delays are accounted for.
- **Q-routing:** Tabular Q-learning (Boyan & Littman 1993). Trained for 200 episodes on the mean traffic matrix with learning rate α = 0.1. Serves as a simpler distributed RL baseline.
- **Random:** Uniform random next-hop selection. Used as a sanity-check upper bound.

---

## Data Format

### Topology file

```
<num_nodes> <num_links>
<u> <v> <distance> <bandwidth> <delay> [...]
...
```

Node indices are 1-based in the file and converted to 0-based internally. Each line after the header describes one undirected link; additional fields beyond `delay` are ignored. Link delays default to 1.0 ms if not specified.

### link_weight.json

An array of floats specifying per-directed-edge delay overrides, in the order they appear in the topology file (each undirected edge u–v yields two entries: u→v then v→u):

```json
[3.0, 9.0, 8.0, 8.0, ...]
```

### Traffic matrices

The loader supports `.dat` (CSV with `#` comments, values in Gbytes/s), `.npy` (NumPy binary), `.txt` (space- or comma-separated), and `.xml` (GEANT format). Values are normalised to integers preserving relative proportions (dividing by the minimum non-zero value, then reducing by GCD), so that each non-zero (source, destination) pair has at least one packet generated per episode.

---

## Configuration Parameters

The `ScaIRConfig` dataclass in [scair/config.py](scair/config.py) exposes all hyperparameters. Values match Table 1 of the paper unless noted otherwise in the section below.

| Parameter | Default | Paper notation |
|---|---|---|
| `feature_length` | 128 | F_l |
| `neural_units` | 64 | N_u |
| `gnn_init_iters` | 8 | K (initialisation iterations) |
| `gnn_update_iters` | 3 | I_n |
| `discount_factor` | 1.0 | γ |
| `learning_rate_initial` | 0.001 | L_r |
| `sigma_initial` | 0.9 | σ initial |
| `sigma_min` | 0.1 | σ minimum |
| `sigma_decrement` | 0.05 | σ decrement |
| `sigma_decay_freq` | 10 | Decay every 10 episodes |
| `memory_size` | 200 | M_s |
| `batch_size` | 64 | B_s |
| `learning_cycle` | 10 | L_c |
| `target_update_freq` | 10 | Hard-copy Q-net → target every 10 episodes |
| `max_episodes` | 400 | Maximum episode |
| `packets_per_episode` | 50 | P_n |
| `generation_interval` | 0.5 ms | G_i |
| `distribution_ratio` | 0.5 | D_r |
| `transmission_time` | 1.0 ms | Fixed link transmission time |
| `queue_time_per_packet` | 1.0 ms | Added per queued packet |
| `max_hops` | 50 | Loop prevention limit |
| `max_nodes` | 30 | One-hot encoding size for destinations and actions |
| `max_degree` | 10 | Padding size for queue and action vectors |
| `action_history_len` | 5 | k (action history entries) |

---

## Experiments

Seven experiments are provided in `experiments/`. See [`results/CONCLUSIONS.md`](results/CONCLUSIONS.md) for full quantitative results and scientific analysis.

### Experiment 1 — D_r Sweep on Abilene (`experiments/comparison_experiment.py`)

Compares four ScaIR variants (per-node/shared × mean/attention) against OSPF across D_r ∈ {0.0, 0.2, 0.4, 0.6, 0.8} using ε-greedy exploration.

**Key result:** +35–57% over OSPF at D_r ≥ 0.6; all four variants are statistically indistinguishable. Outputs: `results/01_dr_comparison/`

### Experiment 2 — Topology Robustness (`experiments/topology_robustness_variants.py`)

Tests all four variants under four topology mutations (add node, remove link, add link, remove node) using a 4-phase A→B→C→D protocol.

**Key result:** Link mutations cause <4% immediate degradation and are fully recovered after 200 adaptation episodes. Node addition is the hardest mutation (+60–70% immediate degradation, partial recovery). Outputs: `results/02_topology_robustness/`

### Experiments 3 & 4 — BRAIN and Germany50 (`experiments/comparison_experiment.py`)

Same D_r sweep with **UCB exploration** on BRAIN (9 nodes) and Germany50 (50 nodes).

```bash
python experiments/comparison_experiment.py \
    --topo data/BRA/Topology.txt --tm_dir data/BRA/TrafficMatrix \
    --results results/03_brain_ucb --action_method ucb

python experiments/comparison_experiment.py \
    --topo data/GER50/Topology.txt --tm_dir data/GER50/TrafficMatrix \
    --results results/04_germany50_ucb --action_method ucb
```

**Key result:** ScaIR beats OSPF at every D_r on both topologies (no crossover, unlike Abilene). Germany50 shows a first clear variant ordering: attention per-node leads at high D_r by ~10%. Outputs: `results/03_brain_ucb/`, `results/04_germany50_ucb/`

### Experiment 5 — No-GNN Ablation (`experiments/ablation_no_gnn.py`)

Replaces the trained SubGNN with fixed (non-learned) feature vectors — one-hot node ID or binary neighbour mask — and compares against full ScaIR. Tests both per-node and shared Q-network configurations. Run on BRAIN, Abilene, and Germany50 with UCB.

**Key result:** Fixed encodings match or outperform the learned GNN within a 300-episode budget on all three topologies. On Germany50, no-GNN variants beat ScaIR-with-GNN by 5–9%. The multi-agent DQN with local observations (not the GNN) is the primary driver of performance. Outputs: `results/05_no_gnn_{bra,abi,ger50}/`

### Experiment 6 — Transfer Learning (`experiments/transfer_experiment.py`)

Copies the trained weights from one Abilene UCB agent (node 0) to all 50 Germany50 agents and fine-tunes for 200 episodes. The Q-network first layer is zero-padded from 218→238 input dimensions, preserving the semantic block structure (destination, queues, GNN feature, action history).

**Key result:** Transfer provides a marginal but consistent head start (1–5% improvement over training from scratch at 200 episodes), suggesting Abilene's routing policy partially generalises across topologies. Outputs: `results/06_transfer/`

### Experiment 7 — Paper f_w vs Our f_w (`experiments/paper_vs_ours_experiment.py`)

Compares the paper's formulation of the GNN update step against our implementation:

- **Paper**: `V_n^(t) = mean_y( f_w(V_y^(t-1)) )` — f_w applied per-neighbour, then averaged; input dim = F_l
- **Ours**: `V_n^(t) = f_w( concat(V_own, mean(V_nbrs)) )` — own state explicitly included; input dim = 2·F_l

Both run on Abilene and GEANT with UCB, per-node weights, g(V_n) fed to the Q-network.

**Key result:** The two formulations are statistically indistinguishable across all D_r values on both topologies (differences < 2%). Including V_own in f_w's input provides no measurable benefit. Outputs: `results/07_paper_vs_ours/`

### Results summary

```
results/
├── CONCLUSIONS.md
├── 01_dr_comparison/      # Exp 1 — Abilene ε-greedy, 4 variants
├── 02_topology_robustness/ # Exp 2 — Abilene topology mutations
├── 03_brain_ucb/          # Exp 3 — BRAIN UCB, 4 variants
├── 04_germany50_ucb/      # Exp 4 — Germany50 UCB, 4 variants
├── 05_no_gnn_bra/         # Exp 5a — BRAIN no-GNN ablation
├── 05_no_gnn_abi/         # Exp 5b — Abilene no-GNN ablation
├── 05_no_gnn_ger50/       # Exp 5c — Germany50 no-GNN ablation
├── 06_transfer/           # Exp 6 — Abilene → Germany50 transfer
└── 07_paper_vs_ours/      # Exp 7 — Paper f_w vs our f_w (Abilene + GEANT)
```

---

## Implementation Notes and Deviations from the Paper

The paper leaves several design choices open or ambiguous. Below are the choices made in this implementation, and where they differ from the paper.

### Simulator

The paper uses OMNet++ for simulation. This implementation uses a self-contained Python event-driven simulator with a heap-based priority queue. The cost model is identical: c_t = (queue length) × 1 ms + 1 ms transmission. The simulation is otherwise faithful to Algorithm 1.

### GNN message-passing aggregation

The paper defines the update `V_n^(t) = f_w^n({V_y^(t-1) : y ∈ N_n})` (Eq. 2) but does not specify the aggregation over the neighbour set. This implementation (`SubGNN`) concatenates the agent's own current feature vector with the **mean** of its neighbours' feature vectors, giving a 2·F_l input to f_w. A paper-faithful alternative (`PaperSubGNN`) applies f_w to each neighbour's feature vector individually and then averages the outputs: `V_n^(t) = mean_y(f_w(V_y^(t-1)))`, with f_w input dim = F_l. Experiment 7 shows the two formulations perform identically in practice.

### Sub-GNN architecture

Both f_w and g_w are two-layer feedforward networks (Linear → ReLU → Linear). They share the same architecture but have independent parameters per agent, consistent with the paper. The initial feature vector is the one-hot node ID zero-padded to F_l (Eq. 4).

### Separate learning rates for Q-network and sub-GNN

The paper trains the Q-network and GNN jointly (§4.3) but specifies a single learning rate. This implementation uses **two separate learning rates**: 0.001 for the Q-network and 0.0001 (10× lower) for the sub-GNN. A shared learning rate destabilises training because the GNN output (used as input to the Q-network) shifts too fast each gradient step, causing the Q-network to additionally chase a moving input distribution on top of the bootstrapping issue inherent in DQN. Reducing the GNN learning rate keeps feature vectors stable between gradient updates.

### Two-stage learning rate schedule

The paper specifies L_r = 0.1 for the first 10 episodes and L_r = 0.001 thereafter. This implementation uses **L_r = 0.001 throughout**. With Q-values in the millisecond scale (5–50 ms) and random network initialisation, a learning rate of 0.1 causes gradient steps large enough to push ReLU activations into the permanently-zero region. Starting at 0.001 avoids dead neurons without requiring the two-stage schedule.

### Replay buffer state storage

The paper stores full state transitions (s_t, c_t, a_t, s_{t+1}, e_t). Because the sub-GNN weights evolve during training, replaying an old state that includes a stale GNN output would corrupt the learning signal. This implementation stores **partial states** (destination one-hot, queue lengths, action history) in the replay buffer, and recomputes the GNN output from current weights at training time. This keeps replayed states consistent with the current network state.

### Estimated subsequent forwarding time e_t

The paper uses the next-hop agent's Q-network to estimate e_t (the expected future delivery time, Eq. 6). This implementation uses the **target network** of the next-hop agent for this estimate rather than the online network, to reduce variance in the training target. This is standard practice in DQN and prevents the target from shifting every gradient step.

### Gradient clipping

Gradient norm is clipped to 1.0 during each training step. The paper does not mention this. It was added to prevent occasional large gradient spikes during the early exploration phase (high σ) when Q-values are poorly initialised.

### Last-hop estimated time e_t

When the chosen next hop is the destination node itself, e_t is set to 0.0 rather than calling `min_q_value` on the destination. Destination nodes never make routing decisions and are therefore never trained; their Q-values remain random. Using those random values as e_t would corrupt the target `y = c + γ·0` for every last-hop transition. Setting e_t = 0.0 at the destination is correct: no further forwarding cost is incurred once the packet arrives.

### Max-hops limit

A hard limit of 50 hops per packet is enforced to prevent routing loops during the early training phase when σ is high and many decisions are random. Packets exceeding this limit are dropped. Without this guard, looping packets inflate the average delivery time indefinitely and dominate the training signal.

### Hot-spot pair

The paper sets the busy ingress as node 0 and the busy egress as the last node (node N-1) for distribution ratio D_r experiments. This convention is used unchanged.

### Action history encoding

The paper includes action history of the past k packets in the state (§4.2), with k = 5, but does not specify the encoding. This implementation encodes each past action as a one-hot vector of length max_degree, and concatenates k such vectors, giving an action history segment of length k × max_degree in the full state vector.
