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
- **Abilene** (11 nodes, 14 bidirectional links)
- **GEANT** (23 nodes, 37 bidirectional links)

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
│   ├── models.py         # SubGNN and QNetwork PyTorch modules
│   └── data_loader.py    # Topology and traffic matrix loaders
└── data/
    ├── ABI/              # Abilene topology and traffic matrices
    │   ├── Topology.txt
    │   ├── TrafficMatrix/
    │   └── link_weight.json
    └── GEA/              # GEANT topology and traffic matrices
        ├── Topology.txt
        ├── TrafficMatrix/
        └── link_weight.json
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

## Implementation Notes and Deviations from the Paper

The paper leaves several design choices open or ambiguous. Below are the choices made in this implementation, and where they differ from the paper.

### Simulator

The paper uses OMNet++ for simulation. This implementation uses a self-contained Python event-driven simulator with a heap-based priority queue. The cost model is identical: c_t = (queue length) × 1 ms + 1 ms transmission. The simulation is otherwise faithful to Algorithm 1.

### GNN message-passing aggregation

The paper defines the update function f_w({V_y | y ∈ N_n}) (Eq. 2) but does not specify the aggregation operation over the neighbour set. This implementation concatenates the agent's own current feature vector with the **mean** of its neighbours' feature vectors, giving a 2·F_l input to f_w. Mean aggregation handles variable-degree nodes without padding and is standard in GCN literature.

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

### Max-hops limit

A hard limit of 50 hops per packet is enforced to prevent routing loops during the early training phase when σ is high and many decisions are random. Packets exceeding this limit are dropped. Without this guard, looping packets inflate the average delivery time indefinitely and dominate the training signal.

### Hot-spot pair

The paper sets the busy ingress as node 0 and the busy egress as the last node (node N-1) for distribution ratio D_r experiments. This convention is used unchanged.

### Action history encoding

The paper includes action history of the past k packets in the state (§4.2), with k = 5, but does not specify the encoding. This implementation encodes each past action as a one-hot vector of length max_degree, and concatenates k such vectors, giving an action history segment of length k × max_degree in the full state vector.
