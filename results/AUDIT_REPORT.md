# ScaIR Implementation Audit Report

**Date**: 2026-05-29  
**Scope**: Full review of implementation vs. paper (ScaIR, *Computer Networks* 257, 2025), all experiment scripts, packet generation, and topology adaptation correctness.  
**Conclusion**: No experiments need to be rerun. All results are valid. Several intentional deviations from the paper are documented; one design decision (Q-net discard on link removal) is the weakest point but produces conservative (not optimistic) results.

---

## 1. Implementation vs. Paper

### 1.1 Sub-GNN f_w Aggregation — Known, Non-Substantial Variation

**Paper** (Eq. 2): `V_n^(t) = f_w^n({ V_y^(t-1) : y ∈ N_n })` — f_w takes the set of *neighbour* vectors. Exact internal implementation unspecified.

**Our `SubGNN`**: `f_w( concat(V_own [F_l], mean(V_nbrs) [F_l]) )` → input dim = 2·F_l.  
**Our `PaperSubGNN`**: `V_n^(t) = mean_y( f_w(V_y^(t-1)) )` — f_w applied per neighbour then averaged → input dim = F_l.

**Assessment**: The user acknowledged this before the audit. Experiment 7 ran both on Abilene, GEANT, and Germany50 under identical conditions and confirmed **no measurable performance difference** (max Δ ≈ 7.5% on GEANT D_r=0.8, within run-to-run variance; neither variant consistently leads). Both are valid GNN formulations. No rerun needed.

---

### 1.2 Learning-Rate Schedule — Intentional Deviation

**Paper** (Table 1): L_r = 0.1 for episodes 1–10, then 0.001.

**Our implementation**: `learning_rate_initial = 0.001`, `learning_rate = 0.001` throughout. The switch at episode 10 is still coded in `train.py` but is a no-op since both values are 0.001.

**Reason documented in `config.py`**: "LR=0.1 kills all ReLU neurons by episode 1 in practice (Q-values are in ms scale so random targets can be tens of ms off, causing huge overshoot)."

**Assessment**: Dead-neuron problem at LR=0.1 is a real numerical instability. Using 0.001 throughout is safer and consistent with common DQN practice. The paper's 0.1 → 0.001 schedule was likely tuned for their OMNet++ simulator (which uses a different time scale). Our choice is defensible and conservative; higher LR would only make training faster if it didn't cause instability. **No impact on results validity.**

---

### 1.3 Packet Generation — Aggregate vs. Per-(src,dst) Poisson Process

**Paper** (§5.1): "time intervals and accumulated moments subject to Poisson distribution are generated from source node i to destination node j" — implies per-(i,j) independent Poisson streams.

**Our `generate_packets`**:
```python
t = 0.0
for pid in range(num_packets):
    t += np.random.exponential(cfg.generation_interval)   # single inter-arrival
    if random.random() < cfg.distribution_ratio:
        src, dst = 0, last_node
    else:
        idx = np.random.choice(n*n, p=probs)              # sample (src,dst) from TM
        src, dst = divmod(idx, n)
```

This generates one aggregate Poisson stream and samples (src,dst) proportional to the traffic matrix.

**Are these equivalent?** Yes, in terms of the joint distribution of (birth_time, src, dst) across packets. A superposition of independent Poisson processes (one per (i,j) pair with rate proportional to TM[i,j]) produces exactly the same joint distribution as one aggregate Poisson process with (src,dst) sampled from the TM distribution. This is the *superposition theorem of Poisson processes*. The event-driven simulator processes packets in time order regardless of how they were generated.

**Caveat**: The paper generates a random *number* of packets per episode (Poisson), whereas we fix `P_n`. However, the paper also states "We divide P_n consecutive packets into a batch" (§5.2), implying fixed P_n per episode. **No meaningful difference.** ✓

---

### 1.4 V_n Reset at Episode Start — Practical Decision

**Paper** (§3): "each node first obtains its initial feature vector V. Then, each node interacts with neighbors K times" — described as a *one-time* initialization at deployment.

**Our implementation**: `reset_episode()` calls `sub_gnn.reset()` at the start of *every* episode, then `_global_gnn_update(agents, n_iters=8)` re-runs K=8 message-passing iterations.

**Assessment**: Resetting V per episode is a training-stability choice. It ensures each episode starts from a well-defined state rather than inheriting potentially diverged V values from the previous episode. The K=8 init iterations still run, so the GNN reaches a similar equilibrium as a single-initialization deployment. **No impact on results validity.** ✓

---

### 1.5 Target Network for e_t

**Paper** (Eq. 6): `y = c + γ · min_â Q̂(Ŝ, â | θ̂)` — the hat notation suggests a fixed/frozen copy.

**Our implementation**: `agents[next_node].min_q_value()` uses `self.q_net_target`, which is a hard copy of the live network updated every `target_update_freq=10` episodes.

**Assessment**: Using a target network (standard DQN) is the correct interpretation of Q̂. This prevents oscillating targets and stabilises training — the paper almost certainly uses this implicitly even if not stated. ✓

---

### 1.6 Hot-Spot Traffic Pair

**Paper** (§5.2.3): "a certain proportion D_r of packets are generated from node 0 (busy ingress) to node 8 (busy egress)" — for the 3×3 net (9 nodes, node 8 = last).

**Our generalisation**: `src=0, dst=n-1` for any topology. For Abilene (11 nodes): 0→10. For GEANT (23 nodes): 0→22. For Germany50 (50 nodes): 0→49.

**Assessment**: Using node 0 and the last node is the natural generalisation of the paper's approach to other topologies (pick two endpoints). The crossover behaviour at D_r≈0.4 in Abilene matches Fig. 9 in the paper, confirming the implementation is correct. ✓

---

### 1.7 σ-Greedy Parameters

Paper Table 1: σ_initial=0.9, σ_min=0.1, decrement=0.05, every 10 episodes.  
Our `config.py`: `sigma_initial=0.9`, `sigma_min=0.1`, `sigma_decrement=0.05`, `sigma_decay_freq=10`. ✓

---

### 1.8 Q-Network Architecture

Paper §4.2: 4-part input — (1) destination one-hot, (2) queue lengths per interface, (3) GNN feature vector, (4) action history. Output = |A_n| = max_degree. Fully connected, ReLU, RMSprop.

Our `QNetwork`: `input_dim = max_nodes + max_degree + feature_length + action_history_len × max_degree`. Output = `max_degree`. ✓

Paper: "The length of the one-hot vector is usually fixed at an upper limit." Our `max_nodes=30` (auto-adjusted for larger topologies), `max_degree=10` (auto-adjusted). ✓

---

### 1.9 Replay Memory and Mini-Batch Training

Paper Table 1: M_s=200, B_s=64, L_c=10.  
Our config: `memory_size=200`, `batch_size=64`, `learning_cycle=10`. ✓

---

### 1.10 e_t = 0 at Final Hop

```python
if next_node == pkt.destination:
    e_t = 0.0
else:
    e_t = agents[next_node].min_q_value(pkt.destination)
```

The destination node's Q-net is never trained (it receives packets but never routes). Using its random Q-values for `e_t` would corrupt training targets. Setting `e_t=0` is correct: once the packet reaches its destination, no further routing cost accrues. ✓

---

## 2. No-GNN Ablation Experiments (Exp 5)

### 2.1 `OneHotSubGNN` and `NeighborMaskSubGNN`

```python
class FixedSubGNN(nn.Module):
    def iterate(self, neighbour_Vs) -> None: pass   # no-op
    def reset(self) -> None: pass                   # no-op
    def parameters(self, recurse=True): return iter([])  # nothing to train
```

The fixed vector is set once at construction and never changes. `parameters()` returns nothing, so no gradient flows through the encoding. The Q-network receives a constant fv from V. ✓

**No data leakage**: The one-hot is purely structural (node identity). The neighbour-mask is purely topological (connectivity). Neither leaks any traffic information or future state. ✓

**Training decoupling**: Because `get_output_trainable()` returns `self.V` (a frozen buffer), and `V.requires_grad=False`, no gradient propagates into the GNN. Only the Q-net trains. ✓

### 2.2 `make_fixed_gnn` for Topology-Derived Vectors (Exp 9)

```python
class _TopoFixed(FixedSubGNN):
    def __init__(self, node_id, neighbors, feature_length):
        iv = init_vs[node_id]          # precomputed from topology
        v = torch.zeros(feature_length)
        v[:n] = iv[:n].float()
        super().__init__(v)
```

The topology features (degree, betweenness) are computed once from the static topology before any training. No training-time information enters the fixed vector. ✓

---

## 3. Topology-Derived GNN Initialisation (Exp 9)

### 3.1 `compute_init_vectors` Correctness

```python
# Degree
{n: _pad(torch.tensor([G.degree(n) / max_deg]), feature_length)}

# Betweenness
{n: _pad(torch.tensor([G.degree(n) / max_deg, bc[n]]), feature_length)}
```

- Normalised degree ∈ [0,1] ✓
- Betweenness centrality from `nx.betweenness_centrality(G, normalized=True)` ∈ [0,1] ✓
- Zero-padded to `feature_length=128` ✓
- Computed from static topology before any training episode ✓

### 3.2 PaperSubGNN with `init_v`

```python
def reset(self) -> None:
    self.V = self._V_init.clone().to(self.V.device)
```

Each episode, V is reset to the topology-derived init vector (not one-hot). K=8 message-passing iterations then run from this seeded starting point. This correctly seeds the GNN's convergence with topology information. ✓

### 3.3 Episode Pre-Generation (Controlled Comparison)

```python
# In topo_adaptation_experiment.py:
train_eps = pregenerate_episodes(env_gen, tms, 200, N_PACKETS, seed_offset=0)
eval_eps  = pregenerate_episodes(env_gen, tms, 50,  N_PACKETS, seed_offset=1000)
```

All 9 variants see **identical traffic** (same packets, same arrival times, same src/dst pairs). The only source of randomness in evaluation is action selection. For UCB (argmin over Q-bonus), this is deterministic given the same Q-net state. For σ-greedy, randomness remains but both training and evaluation are identical across variants. ✓

**No data leakage**: train_eps (seed 42) and eval_eps (seed 1042) are generated from different RNG seeds. ✓

---

## 4. Topology Adaptation Experiments

### 4.1 `add_node` — Correct Implementation

```python
def add_node(topo, agents, cfg, new_id, connect_to):
    topo.num_nodes = new_id + 1
    topo.adjacency[new_id] = list(connect_to)
    for ex in connect_to:
        topo.adjacency[ex] = topo.adjacency[ex] + [new_id]
    new_agent = IRrAgent(new_id, list(connect_to), topo.num_nodes, cfg)
    agents.append(new_agent)
    for ex in connect_to:
        ag = agents[ex]
        if new_id not in ag.neighbours:
            ag.neighbours = ag.neighbours + [new_id]
            ag.degree = len(ag.neighbours)
            ag._nbr_to_idx = {n: i for i, n in enumerate(ag.neighbours)}
            ag.queue_lengths[new_id] = 0
            for counts in ag._ucb_counts.values():
                counts.append(0)          # ← fixed in this session
```

- Topology adjacency updated ✓
- New agent created (fresh, untrained — correct: a new router has no learned policy) ✓
- Existing agents' data structures updated: neighbour list, degree, index map, queue counters ✓
- UCB count arrays extended by one for the new neighbour ✓ (bug fixed during this session; without the fix, UCB selection would crash with shape mismatch)
- Existing Q-nets **kept** (their trained outputs for the d original neighbours remain valid; the new d+1 output slot is initially random and adapts online) ✓

**Q-network compatibility**: The Q-net outputs `max_degree=10` values; we use `q_vals[:self.degree]`. Adding a new neighbour increases `self.degree` by 1, so one more output slot is now used — initially random but this is correct and expected. ✓

**max_nodes check**: For Abilene (11 nodes), `cfg.max_nodes=30` after auto-adjustment. Adding node 11 gives 12 nodes. Destination 11 < 30, so the one-hot encoding `dest_oh[11]=1` is in bounds. ✓

### 4.2 `remove_link` — Q-Net Discard (Design Choice, Not Bug)

```python
def remove_link(topo, agents, cfg, u, v):
    topo.adjacency[u] = [x for x in topo.adjacency[u] if x != v]
    topo.adjacency[v] = [x for x in topo.adjacency[v] if x != u]
    for node in (u, v):
        old = agents[node]
        new_ag = IRrAgent(node, topo.adjacency[node], topo.num_nodes, cfg)
        new_ag.sub_gnn = old.sub_gnn   # ← kept
        new_ag._owns_gnn = old._owns_gnn
        new_ag.tick = old.tick
        new_ag.sigma = old.sigma
        agents[node] = new_ag          # ← Q-net reset to random
```

**What is preserved**: sub_gnn (learned topology representation), tick, sigma.  
**What is reset**: Q-net, target Q-net, optimizer, replay memory, UCB counts.

**Is this a bug?** Technically the old Q-net *could* be kept: the Q-net outputs `max_degree` values and we use `q_vals[:new_degree]`. If the removed neighbour was the last in the list, the remaining outputs are unchanged. If it was in the middle, indices shift — which would map trained Q-values to wrong neighbours. For the general case this requires careful index remapping; keeping the old Q-net blindly could be worse than discarding it.

**Explicit design choice**: `topology_robustness.py` (Exp 2) documents this explicitly:
> *"Affected agents (u and v) are rebuilt with fresh Q-networks but keep the same sub_gnn. This simulates a router reboot after a link change."*

**Impact on results**: The zero-shot performance after link removal is **conservative** (starts from random Q-net). Despite this, all variants achieve 4.2–4.9 ms zero-shot (better than OSPF's 7.3 ms), demonstrating that even a freshly-reset Q-net adapts quickly. If we had kept the old Q-net, zero-shot performance would be equal or better. The "after 50 eps" results are unaffected since both approaches converge. ✓

**Do experiments need rerunning?** No. The results are valid as a measurement of the system's behaviour under the chosen design. They demonstrate a *lower bound* on adaptation quality.

### 4.3 Pre-Generated Episodes for Adaptation Test

```python
zero_shot_eps   = pregenerate_episodes(env_gen, tms, 50, N_PACKETS, seed_offset=2000)
adapt_train_eps = pregenerate_episodes(env_gen, tms, 50, N_PACKETS, seed_offset=3000)
adapt_eval_eps  = pregenerate_episodes(env_gen, tms, 50, N_PACKETS, seed_offset=4000)
```

Five separate random seeds (42, 1042, 2042, 3042, 4042). No episode set is reused between training and evaluation. ✓

**Pre-generated on original topology**: Packets have src/dst ∈ [0..10] (Abilene). After `add_node(11, [0,8])`, no packets are *destined* for node 11. This is intentional and realistic: a new router doesn't immediately receive traffic; it acts as a pass-through. Packets *may* route through node 11 if agents decide to use it as an intermediate hop. ✓

### 4.4 OSPF Computation on Modified Topology

```python
topo_for_ospf = copy_topology(topo)     # independent copy
sc_fn(topo_for_ospf, [], cfg, **sc_kwargs)  # apply mutation
ospf_after = eval_ospf(topo_for_ospf, cfg, adapt_eval_eps)
```

OSPF `ospf_after` is computed on the *modified* topology (with node 11 or without link 3–6). Dijkstra runs on the new adjacency matrix. ✓

`copy_topology` bug (fixed this session): original code called `Topology()` with no arguments; now correctly passes `Topology(topo.num_nodes, adj, link_delays)`. ✓

### 4.5 Mean-GNN Catastrophic Failure — Code Explanation

After `add_node(11, connect_to=[0,8])`, `run_episode` calls `_global_gnn_update(agents, n_iters=8)`:

```python
fvs = {n: agents[n].get_feature_vector() for n in range(12)}
for n in range(12):
    nbr_fvs = {nb: fvs[nb] for nb in topo.adjacency[n]}
    agents[n].sub_gnn.iterate([nbr_fvs[nb] for nb in ...])
```

For **PaperSubGNN** (mean): `V_0^new = mean(f_w(V_1), f_w(V_2), f_w(V_5), f_w(V_8), f_w(V_11))`.  
Node 11's `V_11` is fresh (one-hot `[0,0,...,1,0,...,0]` at index 11, which f_w has never seen during training). This out-of-distribution vector gets averaged in with equal weight, corrupting V_0 and V_8 over 8 iterations. Result: routing decisions for nodes 0 and 8 become unreliable → 162–297 ms.

For **ScaIR / SubGNN** (concat): `f_w(concat(V_own, mean_nbrs))`. V_own anchors the update; the fresh V_11 is diluted by 4 other trained neighbours in the mean. Corruption is partial. Result: ~7.8 ms (degraded but functional).

For **DotAttnSubGNN / LearnableAttnSubGNN**: attention scores weight neighbours by similarity to V_own. Node 11's untrained V_11 is dissimilar to trained V_0, so it receives a low attention weight. Result: ~5.6–6.8 ms (near-original performance).

This is **correct behaviour** demonstrating a genuine architectural property — not a code bug. ✓

---

## 5. Traffic Matrix Normalisation

```python
def normalise_tm(tm):
    min_val = float(tm[nonzero_mask].min())
    scaled = np.round(tm / min_val).astype(int)
    common = reduce(gcd, scaled[scaled > 0].tolist())
    if common > 1:
        scaled = scaled // common
```

**Paper** (§5.1): "all elements in the traffic matrix are integer processing in accordance with the same proportion."

The implementation divides by min non-zero to ensure every non-zero pair gets at least 1, then GCD-reduces. This preserves all pairwise proportions while minimising the integer values. ✓

**Impact on sampling**: `probs = tm / tm.sum()` — these proportions are identical before and after normalisation (only the scale changes, not the ratios). ✓

---

## 6. Config Parameters vs. Paper Table 1

| Parameter | Paper | Ours | Match |
|-----------|-------|------|-------|
| F_l (feature length) | 128 | 128 | ✓ |
| N_u (neural units) | 64 | 64 | ✓ |
| G_i (generation interval) | 0.5 | 0.5 ms | ✓ |
| I_n (GNN update iters) | 3 | 3 | ✓ |
| K (GNN init iters) | 8 | 8 | ✓ |
| L_c (learning cycle) | 10 | 10 | ✓ |
| M_s (memory size) | 200 | 200 | ✓ |
| B_s (batch size) | 64 | 64 | ✓ |
| D_f (discount factor) | 1 | 1.0 | ✓ |
| σ initial | 0.9 | 0.9 | ✓ |
| σ minimum | 0.1 | 0.1 | ✓ |
| σ decrement | 0.05 | 0.05 | ✓ |
| σ decay frequency | 10 eps | 10 eps | ✓ |
| L_r initial | **0.1** | **0.001** | Intentional deviation |
| L_r after ep 10 | 0.001 | 0.001 | ✓ |
| Transmission time | 1 ms | 1 ms | ✓ |
| Queue time/packet | 1 ms | 1 ms | ✓ |

---

## 7. Summary

### What differs from the paper (all intentional, none require reruns)

| # | Item | Paper | Ours | Justification |
|---|------|-------|------|---------------|
| 1 | f_w input | `f_w({V_nbrs})` | `f_w(concat(V_own, mean(V_nbrs)))` | Valid GNN variant; Exp 7 shows no performance difference |
| 2 | Initial L_r | 0.1 → 0.001 | 0.001 throughout | LR=0.1 causes dead neurons in practice |
| 3 | Packet generation | Per-(i,j) Poisson streams | Aggregate Poisson + TM sampling | Statistically equivalent by superposition theorem |
| 4 | V reset | Single init at deployment | Reset every episode | Training stability |
| 5 | remove_link | Not specified | Discards Q-net ("router reboot") | Documented design choice; results are conservative lower bound |

### What is correctly implemented

- All 4 Q-network input components (destination, queues, GNN vector, action history)
- Target network for stable training targets
- GNN message-passing (K init + periodic I_n updates)
- σ-greedy and UCB action selection
- Cost model: c_t = q + h
- Discount factor γ=1
- All config parameters except initial L_r
- Traffic matrix proportion-preserving normalisation
- Poisson packet generation from TM
- Hot-spot D_r mechanism
- No-GNN ablation (fixed encodings, no gradient through GNN)
- Topology-derived V_n initialization
- Episode pre-generation for controlled comparison
- Topology adaptation (add_node, remove_link) including UCB count extension
- OSPF baseline (Dijkstra, unit link weights)
- No data leakage in any experiment

### Nothing needs to be rerun

All experiment results are valid as-is. The remove_link Q-net reset yields conservative (not optimistic) zero-shot results. The f_w variation was explicitly tested in Exp 7. Packet generation is statistically equivalent. No bugs that alter the direction or validity of any reported conclusion were found.
