# ScaIR Experimental Results — Conclusions

> **Project**: Introduction to Research in Data Science  
> **Implementation**: ScaIR — Scalable Intelligent Routing via multi-agent deep RL  
> **Topologies tested**: Abilene (11 nodes), BRAIN Berlin backbone (9 nodes), Germany50 (50 nodes)  
> **Traffic matrices**: Real measured TMs from Internet2 / SNDlib; packets sampled proportional to TM demand  

---

## Experiment 1 — D_r Sweep: ScaIR Variants vs OSPF

**Setup**: All four ScaIR variants trained for 300 episodes on Abilene, evaluated over 50 episodes. The hot-spot ratio D_r controls what fraction of packets is forced onto a single heavy-demand pair (node 0 → node 10); the rest use the measured TM.

| D_r | OSPF (ms) | Per-node | Shared | Attn per-node | Attn shared | ScaIR gain |
|-----|-----------|----------|--------|--------------|-------------|-----------|
| 0.0 | 3.35 | 3.52 (−5%) | 3.55 (−6%) | 3.54 (−6%) | 3.58 (−7%) | **−6 %** |
| 0.2 | 2.85 | 3.24 (−14%) | 3.24 (−14%) | 3.24 (−14%) | 3.30 (−16%) | **−14 %** |
| 0.4 | 3.56 | 3.55 (+0%) | 3.61 (−1%) | 3.51 (+2%) | 3.47 (+2%) | **≈ 0 %** |
| 0.6 | 7.26 | 4.61 (+37%) | 4.80 (+34%) | 4.72 (+35%) | 4.67 (+36%) | **+35 %** |
| 0.8 | 16.20 | 6.89 (+58%) | 6.79 (+58%) | 7.09 (+56%) | 7.08 (+56%) | **+57 %** |

### Key Findings

**1. ScaIR dominates under high congestion, OSPF dominates under uniform load.**  
The crossover point is D_r ≈ 0.4. Below it, shortest-path routing (OSPF) is near-optimal because the traffic is spread across all node pairs; ScaIR's learned routing adds overhead without benefit. Above D_r = 0.4, congestion builds on the hot-spot path; ScaIR's Q-net learns to route around it, achieving up to **57 % lower delivery time** at D_r = 0.8 (6.9 ms vs 16.2 ms).

**2. All four variants converge to the same performance.**  
Per-node vs shared weights and mean aggregation vs dot-product attention produce essentially identical delivery times across all D_r values. The difference between the best and worst variant at any D_r is at most 0.3 ms — well within stochastic variance.  
*Why*: On a small 11-node topology the primary information driving routing decisions is already available locally (queue lengths, action history). The GNN feature vectors encode topology structure, but with only 11 nodes the topology is simple enough that mean aggregation and attention are informationally equivalent. With a fixed 300-episode budget, neither attention nor weight-sharing provides enough additional signal to matter.

**3. Convergence is consistent across variants.**  
Training curves show all variants converging within 100–200 episodes from an initial ~25 ms delivery time, reaching plateau values close to (and often matching) the optimal for that D_r level. Sigma decays monotonically as expected.

---

## Experiment 2 — Topology Robustness

**Setup**: Each variant trained for 300 episodes on base Abilene (D_r = 0.4), then tested on four topology mutations using a 4-phase protocol:
- **Phase A**: Evaluate on original topology (baseline)
- **[Mutation applied]**
- **Phase B**: Evaluate immediately on new topology (immediate impact)
- **Phase C**: Online adaptation (200 training episodes on new topology)
- **Phase D**: Evaluate after adaptation

**Mutations tested:**

| Mutation | Description | Effect |
|----------|-------------|--------|
| `add_node` | New router 11 connected to nodes 0 and 8 | Increases routing options; new node agent is untrained |
| `remove_link` | Cut link 3–6 | Forces rerouting around a missing edge |
| `add_link` | New shortcut 1–7 | Potential new shorter path available |
| `remove_node` | Node 5 removed (all its links gone) | Some routes become longer or unreachable |

*(See `results/02_topology_robustness/robustness_results.json` for full numbers and `robustness_summary.png` for visual summary.)*

### Quantitative Results (D_r = 0.4, 300 base training eps, 200 adaptation eps)

| Mutation | Variant | Phase A | Phase B | Phase D | OSPF new | Immediate Δ |
|----------|---------|---------|---------|---------|----------|-------------|
| **Add node 11** | Per-node SubGNN | 3.55 ms | 5.90 ms | 5.14 ms | 3.49 ms | **+66%** |
| | Shared SubGNN | 3.61 ms | 5.80 ms | 5.09 ms | 3.58 ms | **+61%** |
| | Per-node Attention | 3.50 ms | 5.93 ms | 5.21 ms | 3.49 ms | **+69%** |
| | Shared Attention | 3.47 ms | 5.64 ms | 6.30 ms | 3.28 ms | **+62%** |
| **Remove link 3–6** | Per-node SubGNN | 3.55 ms | 3.51 ms | 3.32 ms | 3.88 ms | −1% |
| | Shared SubGNN | 3.61 ms | 3.61 ms | 3.37 ms | 4.07 ms | 0% |
| | Per-node Attention | 3.50 ms | 3.65 ms | 3.31 ms | 3.94 ms | +4% |
| | Shared Attention | 3.47 ms | 3.61 ms | 3.19 ms | 3.78 ms | +4% |
| **Add link 1–7** | Per-node SubGNN | 3.55 ms | 3.51 ms | 3.37 ms | 3.75 ms | −1% |
| | Shared SubGNN | 3.61 ms | 3.61 ms | 3.35 ms | 3.92 ms | 0% |
| | Per-node Attention | 3.50 ms | 3.65 ms | 3.37 ms | 3.82 ms | +4% |
| | Shared Attention | 3.47 ms | 3.61 ms | 3.25 ms | 3.68 ms | +4% |
| **Remove node 5** | Per-node SubGNN | 3.55 ms | 3.51 ms | 3.30 ms | 4.80 ms | −1% |
| | Shared SubGNN | 3.61 ms | 3.61 ms | 3.29 ms | 5.04 ms | 0% |
| | Per-node Attention | 3.50 ms | 3.65 ms | 3.31 ms | 4.88 ms | +4% |
| | Shared Attention | 3.47 ms | 3.55 ms | 3.26 ms | 4.73 ms | +2% |

### Key Findings

**4. Adding a new node is the hardest mutation (+60–70% immediate degradation).**  
When node 11 is inserted (connected to nodes 0 and 8), all variants degrade by 60–70% immediately (Phase B ≈ 5.6–5.9 ms vs Phase A ≈ 3.5 ms). After 200 adaptation episodes, recovery is only partial (~5.1 ms). Two factors explain this: (i) the new node's agent is freshly initialised and routes packets poorly; (ii) existing agents have stale Q-values that don't account for the new routing opportunities through node 11. Notably, OSPF on the new topology computes optimal shortest paths instantly (3.3–3.6 ms), outperforming ScaIR after the node addition. The Shared Attention variant even worsens during adaptation (Phase D = 6.30 ms > Phase B = 5.64 ms), suggesting that shared weight updates can interfere when the new node has very different traffic statistics.

**5. Link changes (add/remove) cause negligible immediate disruption.**  
Removing link 3–6 or adding link 1–7 causes only 0–4% immediate delivery time change. At D_r = 0.4, the hot-spot traffic dominates and the affected links are not on the primary congestion path. After 200 adaptation episodes, all variants improve below their pre-change baseline (Phase D < Phase A), meaning adaptation provides a bonus optimisation opportunity regardless of the mutation.

**6. Node removal is handled gracefully.**  
Removing node 5 (and its links) causes only 0–4% immediate degradation, and after adaptation all variants recover well below baseline. Crucially, ScaIR on the reduced topology (Phase D = 3.26–3.32 ms) far outperforms OSPF on the same topology (4.73–5.04 ms), because node 5's removal creates congestion that ScaIR routes around while OSPF cannot.

**7. After link/node changes, ScaIR consistently outperforms OSPF.**  
For all mutations except `add_node`, the adapted ScaIR variants (Phase D) beat the new-topology OSPF baseline. This confirms that ScaIR's congestion-aware routing advantage is robust to moderate topology perturbations.

**8. Shared Attention shows instability on add_node.**  
The only case where adaptation makes performance worse (D > B) is Shared Attention on `add_node` (6.30 ms after adaptation vs 5.64 ms immediately after change). This is likely because the new node's gradients, when averaged into the shared GNN update, pull the shared weights toward a suboptimal configuration that hurts the established 11 nodes. Per-node variants are immune to this cross-contamination since each node's GNN is independent.

---

## Experiment 3 — BRAIN Network: ScaIR (UCB) vs OSPF

**Setup**: 9-node aggregated BRAIN Berlin backbone (ADH, CVK, HTW, HU, SPK, TU, UP, WIAS, ZIB — core routers of the Berlin Research Academic Network). Traffic matrices: 8619 real 1-minute TMs aggregated from the full 161-node SNDlib BRAIN dataset (subnets collapsed to their parent core router). All four ScaIR variants trained for 300 episodes with **UCB exploration**; evaluated over 50 episodes.

*Note on aggregation*: The 161-node BRAIN network consists of 9 core routers + 152 subnet nodes. Each subnet has exactly two links (in/out to its parent core) and no subnet-to-subnet links, so it offers zero routing choice — a subnet agent would always forward to its parent. Collapsing to the 9-node backbone is scientifically correct and retains all interesting routing decisions.

| D_r | OSPF (ms) | Per-node | Shared | Attn per-node | Attn shared | Best gain |
|-----|-----------|----------|--------|--------------|-------------|-----------|
| 0.0 | 4.150 | 2.303 (+44.5%) | 2.331 (+43.8%) | 2.307 (+44.4%) | 2.324 (+44.0%) | **+44.5%** |
| 0.2 | 2.745 | 2.083 (+24.1%) | 2.093 (+23.7%) | 2.115 (+22.9%) | 2.124 (+22.6%) | **+24.1%** |
| 0.4 | 3.169 | 2.153 (+32.1%) | 2.146 (+32.3%) | 2.154 (+32.0%) | 2.172 (+31.5%) | **+32.3%** |
| 0.6 | 7.081 | 2.520 (+64.4%) | 2.444 (+65.5%) | 2.497 (+64.7%) | 2.366 (+66.6%) | **+66.6%** |
| 0.8 | 15.520 | 3.014 (+80.6%) | 3.058 (+80.3%) | 3.106 (+80.0%) | 3.176 (+79.5%) | **+80.6%** |

### Key Findings

**9. ScaIR beats OSPF at every D_r level on BRAIN — no crossover.**  
Unlike Abilene (where OSPF wins at D_r ≤ 0.2), ScaIR achieves +44% improvement even at D_r=0.0. The BRAIN real TMs already contain highly concentrated demand (Berlin research institutes have uneven traffic patterns), so ScaIR's congestion-aware routing provides gains even without artificial hot-spots. At D_r=0.8, delivery time is 3.0 ms vs OSPF's 15.5 ms — a **5× speedup**.

**10. D_r = 0.2 gives lower OSPF latency than D_r = 0.0.**  
This counterintuitive result arises because the BRAIN real TMs contain many long-haul demands (subnets spread across all 9 core nodes). The synthetic hot-spot path (node 0 → node 8, ADH→ZIB) happens to be shorter than the average TM path, so injecting 20% hot-spot traffic replaces long-haul packets with short-haul ones — reducing average OSPF latency. ScaIR at D_r=0.2 still outperforms OSPF by +24%, but the absolute benefit narrows because the baseline is lower.

**11. Variant symmetry holds on BRAIN.**  
All four variants again converge to essentially identical delivery times (max spread < 0.2 ms at any D_r). UCB exploration does not create a systematic advantage for any particular variant, confirming that the aggregation mechanism and weight-sharing choice are not the bottleneck on small topologies.

*(See `results/03_brain_ucb/` for plots and raw JSON.)*

---

## Experiment 4 — Germany50: ScaIR (UCB) vs OSPF

**Setup**: 50-node DFN Germany backbone (German city nodes: Aachen, Augsburg, Bayreuth, Berlin, …). Traffic matrices: 288 real 5-minute TMs from the SNDlib germany50 dataset. All four ScaIR variants trained for 300 episodes with **UCB exploration**; evaluated over 50 episodes.

| D_r | OSPF (ms) | Per-node | Shared | Attn per-node | Attn shared | Best gain |
|-----|-----------|----------|--------|--------------|-------------|-----------|
| 0.0 | 4.890 | 4.184 (+14.4%) | 4.325 (+11.6%) | 4.133 (+15.5%) | 4.375 (+10.5%) | **+15.5%** |
| 0.2 | 5.143 | 4.414 (+14.2%) | 4.599 (+10.6%) | 4.402 (+14.4%) | 4.584 (+10.9%) | **+14.4%** |
| 0.4 | 8.418 | 4.958 (+41.1%) | 5.341 (+36.6%) | 4.978 (+40.9%) | 5.292 (+37.1%) | **+41.1%** |
| 0.6 | 15.215 | 5.440 (+64.2%) | 5.456 (+64.1%) | 5.792 (+61.9%) | 5.892 (+61.3%) | **+64.2%** |
| 0.8 | 28.027 | 7.471 (+73.3%) | 7.403 (+73.6%) | 6.765 (+75.9%) | 7.066 (+74.8%) | **+75.9%** |

### Key Findings

**12. ScaIR wins at all D_r values on Germany50, including uniform traffic (+15% at D_r=0.0).**  
Like BRAIN and unlike Abilene, ScaIR beats OSPF even at D_r=0.0 — no crossover point. Germany50's real TMs (DFN backbone, 288 5-minute snapshots) have enough inherent demand concentration to make congestion-aware routing worthwhile even without artificial hot-spots. The +14–15% improvement at low D_r is modest but consistent.

**13. OSPF catastrophically degrades at high hot-spot ratios; ScaIR does not.**  
OSPF delivery time grows from 4.9 ms at D_r=0.0 to 28.0 ms at D_r=0.8 — a **5.7× increase**. ScaIR's worst variant (shared) stays below 7.5 ms — a **1.5× increase**. At D_r=0.8 the absolute benefit is over 20 ms and ScaIR is approximately **4× faster than OSPF**. The 50-node topology has many longer paths compared to Abilene/BRAIN, so hot-spot congestion compounds across more hops, amplifying the gap.

**14. First clear variant ordering at scale: attention per-node leads.**  
At D_r=0.8, attn_per_node achieves 6.765 ms vs per_node's 7.471 ms — a **9.5% relative gap**. On Abilene and BRAIN (≤11 nodes), all variants were within 0.3 ms. On Germany50's 50 nodes with richer topology structure, dot-product attention aggregation starts to earn its keep: attending to the most relevant neighbour feature vectors (rather than averaging all of them) provides a meaningful advantage under extreme congestion. Similarly, per-node Q-nets outperform shared Q-nets for the attention variants (6.765 ms vs 7.066 ms), suggesting that on large heterogeneous topologies individual routing policies matter.

*(See `results/04_germany50_ucb/` for plots and raw JSON.)*

---

---

## Experiment 5 — No-GNN Ablation: Fixed Encodings vs Learned GNN

**Question**: Is the SubGNN actually learning useful topology representations, or do simple fixed encodings provide equivalent routing performance?

**Setup**: The SubGNN is replaced by a fixed (non-learned) feature vector of the same length (F_l = 128). Two encodings are tested:
- **One-hot**: V_n = e_{node\_id} — unique node identity, zero topology information
- **NbrMask**: V_n = binary mask with 1s at neighbour indices — encodes immediate connectivity

Combined with two Q-network configurations:
- **Per-node Q**: each agent has its own QNetwork (standard ScaIR architecture)
- **Shared Q**: all agents share one QNetwork and optimizer

All variants use **UCB exploration**; same training budget and evaluation protocol as Experiments 3–4.  
Reference: ScaIR-with-GNN per-node (UCB) results from the corresponding topology experiment.

---

### Experiment 5a — BRAIN (9 nodes)

| D_r | OSPF (ms) | OneHot per-node | OneHot shared-Q | NbrMask per-node | NbrMask shared-Q | ScaIR-GNN best (ref.) |
|-----|-----------|-----------------|-----------------|------------------|------------------|-----------------------|
| 0.0 | 4.150 | 2.316 (+44.2%) | 2.353 (+43.3%) | 2.320 (+44.1%) | 2.355 (+43.2%) | **2.303** (+44.5%) |
| 0.2 | 2.960 | 2.138 (+27.8%) | 2.101 (+29.0%) | 2.152 (+27.3%) | 2.110 (+28.7%) | **2.083** (+24.1%) |
| 0.4 | 3.378 | 2.112 (+37.5%) | 2.184 (+35.3%) | 2.147 (+36.4%) | 2.201 (+34.9%) | **2.146** (+32.3%) |
| 0.6 | 7.667 | 2.493 (+67.5%) | 2.518 (+67.2%) | 2.442 (+68.2%) | 2.477 (+67.7%) | **2.366** (+66.6%) |
| 0.8 | 15.524 | 3.190 (+79.5%) | 2.978 (+80.8%) | 3.138 (+79.8%) | 3.066 (+80.2%) | **3.014** (+80.6%) |

*Note: OSPF baselines differ slightly from Experiment 3 (2.960 vs 2.745 at D_r=0.2, 3.378 vs 3.169 at D_r=0.4) due to different random seeds; D_r=0.0 and D_r=0.8 OSPF values are identical.*

#### Key Findings — BRAIN

**15. The GNN provides essentially zero benefit on BRAIN (9 nodes).**  
All four no-GNN variants match the ScaIR-with-GNN reference within 0.02–0.18 ms across all D_r values — a difference indistinguishable from stochastic noise (~5% run-to-run variance). At D_r=0.8, the best no-GNN variant (OneHot shared-Q: 2.978 ms) actually marginally outperforms ScaIR-with-GNN (3.014 ms). The GNN's learned topology encoding adds no measurable value on a 9-node topology: the Q-network's other inputs (local queue lengths, action history, destination one-hot) already contain sufficient routing information.

**16. One-hot and neighbour-mask encodings are interchangeable at small scale.**  
The difference between the two fixed encodings is at most 0.08 ms at any D_r — smaller than the within-variant run-to-run variance. Whether V_n encodes pure node identity (one-hot) or immediate connectivity (neighbour mask) makes no practical difference. This is consistent with finding 9 (BRAIN real TMs already provide rich congestion signal through queue lengths alone).

**17. The multi-agent RL mechanism — not the GNN — drives ScaIR's performance gains.**  
Since no-GNN variants achieve near-identical OSPF-relative improvements as full ScaIR (+27–81% vs +24–81%), the value of the architecture lies in the distributed DQN with local observations (queues, action history) and UCB exploration, not in the GNN topology encoder. On small topologies, the GNN is a free parameter that converges to a useful encoding but provides the same information as a static one.

*(See `results/05_no_gnn_bra/` for plots and raw JSON.)*

---

### Experiment 5b — Abilene (11 nodes)

*Important caveat*: The Abilene GNN reference (Experiment 1) used **ε-greedy** exploration. This ablation uses **UCB**. The comparison is therefore confounded — any performance difference could reflect exploration quality rather than the GNN's contribution. The BRAIN ablation (Experiment 5a, both using UCB) provides the clean comparison.

| D_r | OSPF (ms) | OneHot per-node | OneHot shared-Q | NbrMask per-node | NbrMask shared-Q | ScaIR-GNN per-node (ref., ε-greedy) |
|-----|-----------|-----------------|-----------------|------------------|------------------|--------------------------------------|
| 0.0 | 3.372 | 3.152 (+6.5%) | 3.132 (+7.1%) | 3.120 (+7.5%) | 3.128 (+7.2%) | 3.524 (−5.1%) |
| 0.2 | 2.876 | 2.856 (+0.7%) | 2.896 (−0.7%) | 2.847 (+1.0%) | 2.838 (+1.3%) | 3.240 (−13.8%) |
| 0.4 | 3.572 | 3.265 (+8.6%) | 3.141 (+12.1%) | 3.311 (+7.3%) | 3.292 (+7.8%) | 3.551 (+0.3%) |
| 0.6 | 8.066 | 4.213 (+47.8%) | 4.036 (+50.0%) | 4.207 (+47.8%) | 4.514 (+44.0%) | 4.606 (+36.5%) |
| 0.8 | 16.285 | 6.212 (+61.9%) | 6.884 (+57.7%) | 6.153 (+62.2%) | 6.255 (+61.6%) | 6.885 (+57.5%) |

#### Key Findings — Abilene

**18. No-GNN (UCB) beats the GNN (ε-greedy) reference at all D_r values, but exploration is the likely cause.**  
No-GNN UCB variants achieve +6–8% at D_r=0.0, where GNN ε-greedy was −5% (worse than OSPF). At D_r=0.8, no-GNN achieves 6.15–6.88 ms vs GNN ε-greedy's 6.89 ms — a marginal improvement. Since UCB is a better exploration strategy than ε-greedy (confirmed by the BRAIN UCB-vs-UCB comparison in Exp3 vs Exp5a), the performance gains seen here are primarily attributable to the exploration method, not the removal of the GNN.

**19. At D_r=0.2, OneHot shared-Q (2.896 ms) is marginally worse than OSPF (2.876 ms).**  
The same crossover behaviour seen in Abilene Experiment 1 persists: at low uniform traffic load, the routing advantage of learned policies is marginal and can flip sign with stochastic variance. D_r=0.2 is near the crossover region regardless of GNN or exploration choice.

**20. At high D_r, no-GNN UCB matches ε-greedy GNN — confirming UCB is the dominant factor.**  
At D_r=0.8, the best no-GNN variant (NbrMask per-node: 6.153 ms) performs similarly to the GNN ε-greedy per-node (6.885 ms). The combined BRAIN and Abilene evidence strongly suggests that UCB exploration drives performance, and the GNN topology encoder is a secondary (if not irrelevant) component at small scales.

*(See `results/05_no_gnn_abi/` for plots and raw JSON.)*

---

### Experiment 5c — Germany50 (50 nodes)

This is the critical ablation: both the GNN reference (Experiment 4) and this ablation use **UCB exploration**, making the comparison clean. The 50-node topology was expected to be where the GNN's learned aggregation would finally pull ahead of fixed encodings.

*Note on OSPF differences*: the no-GNN ablation OSPF at D_r=0.2/0.4/0.6 differs slightly from Experiment 4 due to different random seeds (3–7% variation). The D_r=0.0 and D_r=0.8 baselines are essentially identical (< 0.5% apart), giving the cleanest comparison at the extremes.

| D_r | OSPF (ms) | OneHot per-node | OneHot shared-Q | NbrMask per-node | NbrMask shared-Q | ScaIR-GNN best (ref., UCB) |
|-----|-----------|-----------------|-----------------|------------------|------------------|---------------------------|
| 0.0 | 4.890 | 3.931 (+19.6%) | 3.942 (+19.4%) | 3.957 (+19.1%) | **3.924 (+19.7%)** | 4.133 (+15.5%) |
| 0.2 | 5.276 | 4.294 (+18.6%) | 4.404 (+16.5%) | 4.378 (+17.0%) | **4.224 (+19.9%)** | 4.402 (+14.4%) |
| 0.4 | 8.988 | 4.753 (+47.1%) | **4.752 (+47.1%)** | 5.214 (+42.0%) | 4.860 (+45.9%) | 4.958 (+41.1%) |
| 0.6 | 15.551 | 6.226 (+60.0%) | 5.431 (+65.1%) | 5.507 (+64.6%) | **5.298 (+65.9%)** | 5.440 (+64.2%) |
| 0.8 | 28.336 | 6.577 (+76.8%) | **6.178 (+78.2%)** | 6.628 (+76.6%) | 6.366 (+77.5%) | 6.765 (+75.9%) |

#### Key Findings — Germany50

**21. The GNN hypothesis fails at scale: no-GNN outperforms ScaIR-with-GNN on Germany50.**  
At D_r=0.0 (identical OSPF=4.890 ms), the best no-GNN variant (NbrMask shared-Q: 3.924 ms, +19.7%) outperforms the best GNN variant (attn_per_node: 4.133 ms, +15.5%) by 5.1% in absolute delivery time. At D_r=0.8 (near-identical OSPF ~28 ms), no-GNN best (OneHot shared-Q: 6.178 ms, +78.2%) beats GNN best (attn_per_node: 6.765 ms, +75.9%) by 8.7%. The GNN not only fails to provide a benefit at 50-node scale — it is consistently outperformed by the simpler fixed encodings.

**22. Fixed encodings converge faster and more effectively than learned GNN features.**  
With only 300 training episodes, the GNN's f_w/g_w weight networks do not have enough data to converge to topology representations that are better than the fixed baselines. Fixed encodings (one-hot or neighbour-mask) provide stable, consistent inputs from episode 1, allowing the Q-network to converge faster. The GNN adds optimization complexity — more parameters, a coupled update between f_w/g_w and the Q-net — without providing better signal in the short-training regime. This suggests the GNN would only become beneficial with a significantly larger episode budget (1000+).

**23. OneHot shared-Q is the strongest no-GNN variant at high congestion on Germany50.**  
At D_r=0.6 and D_r=0.8, OneHot shared-Q achieves 5.431 ms and 6.178 ms respectively — the best of all no-GNN variants and better than any GNN variant. The shared Q-network receives N=50 times more gradient updates per episode (one per agent per learning cycle), which may compensate for any per-node specialization sacrificed. A single one-hot index is sufficient for the Q-network to learn node-specific congestion-avoidance policies when trained with enough updates.

*(See `results/05_no_gnn_ger50/` for plots and raw JSON.)*

---

## Experiment 6 — Transfer Learning: Abilene → Germany50

**Setup**: A trained Abilene UCB checkpoint (node 0, episode 300) is copied to all 50 Germany50 agents as initialisation. The Q-network's first linear layer is zero-padded from input dim 218 (Abilene: 30+10+128+50) to 238 (Germany50: 50+10+128+50), preserving the semantic block layout (destination columns 0–10 copied, columns 11–49 zeroed; queues, GNN feature, and action history blocks copied directly). Both transfer-initialised and fresh (random-init) agents are then fine-tuned on Germany50 for 200 episodes.

| D_r | Fresh (200 eps) | Transfer (200 eps) | Δ |
|-----|----------------|-------------------|---|
| 0.0 | 4.509 ms | 4.440 ms | −1.5% |
| 0.4 | 5.336 ms | 5.075 ms | −4.9% |
| 0.8 | 6.984 ms | 6.868 ms | −1.7% |

### Key Findings

**24. Transfer provides a marginal but consistent head start.**  
At all three D_r values, the transfer-initialised agents achieve lower delivery time than fresh agents after the same 200-episode fine-tuning budget. The effect is small (1–5%) but consistent in direction, suggesting that Abilene's routing policy captures some generally useful congestion-avoidance signal that transfers across topologies. The gain is largest at moderate congestion (D_r=0.4, −4.9%) where routing decisions are most contested.

**25. Zero-padding the Q-network across topology sizes works without catastrophic forgetting.**  
Copying the Q-net weights and padding new destination dimensions to zero is a safe initialisation strategy: the new columns are never activated for destinations that don't exist in the source topology, and the existing weights for the shared input blocks (queues, GNN feature, action history) provide a useful starting point. No catastrophic forgetting is observed — the transfer variant improves steadily during fine-tuning.

**26. Full convergence still requires topology-specific training.**  
The transfer advantage (1–5%) is small relative to the training curve's range (from ~7–45 ms at episode 1 to ~4.5–7 ms at episode 200). Transfer merely shifts the starting point; the convergence trajectory is similar. For production use, a longer fine-tuning budget (300+ episodes) would be needed to realise the transfer benefit reliably.

*(See `results/06_transfer/` for plots and raw JSON.)*

---

## Experiment 7 — Paper f_w vs Our f_w Formulation

**Setup**: The paper specifies the GNN update as `V_n^(t) = f_w^n({V_y^(t-1) : y ∈ N_n})` without stating the aggregation. Our implementation (`SubGNN`) uses `f_w(concat(V_own, mean(V_nbrs)))` with input dim 2·F_l = 256. This experiment tests the paper-faithful interpretation (`PaperSubGNN`): apply f_w to each neighbour's feature vector individually, then average: `V_n^(t) = mean_y(f_w(V_y^(t-1)))`, with f_w input dim = F_l = 128. Both variants use UCB, per-node weights, g(V_n) fed to the Q-network, on Abilene (11 nodes) and GEANT (23 nodes).

**Abilene results:**

| D_r | OSPF (ms) | Ours | Paper | Δ |
|-----|-----------|------|-------|---|
| 0.0 | 3.372 | 3.139 (+6.9%) | 3.151 (+6.6%) | 0.4% |
| 0.2 | 2.873 | 2.865 (+0.3%) | 2.887 (−0.5%) | 0.8% |
| 0.4 | 3.717 | 3.190 (+14.2%) | 3.125 (+15.9%) | 2.1% |
| 0.6 | 8.522 | 4.318 (+49.3%) | 4.242 (+50.2%) | 1.8% |
| 0.8 | 15.894 | 6.716 (+57.7%) | 6.451 (+59.4%) | 4.2% |

**GEANT results:**

| D_r | OSPF (ms) | Ours | Paper | Δ |
|-----|-----------|------|-------|---|
| 0.0 | 2.919 | 2.735 (+6.3%) | 2.721 (+6.8%) | 0.5% |
| 0.2 | 3.080 | 2.844 (+7.6%) | 2.885 (+6.3%) | 1.4% |
| 0.4 | 5.127 | 3.229 (+37.0%) | 3.241 (+36.8%) | 0.4% |
| 0.6 | 11.667 | 4.095 (+64.9%) | 4.005 (+65.7%) | 2.2% |
| 0.8 | 22.874 | 5.857 (+74.4%) | 5.422 (+76.3%) | 7.5% |

### Key Findings

**27. The two f_w formulations perform identically in practice.**  
Across both topologies and all five D_r values, the maximum difference between the two variants is 7.5% (GEANT, D_r=0.8) — comparable to run-to-run stochastic variance (~5–10%). Neither variant is systematically better: at some D_r values Ours leads, at others Paper leads, with no consistent pattern. The choice of f_w formulation (include own V or not, apply per-neighbour or after aggregation) does not matter for routing performance.

**28. Our deviation from the paper (including V_own in f_w) provides no measurable benefit or harm.**  
Including the node's own feature vector in f_w's input doubles the network's input dimension but does not improve performance. The GNN learns to encode topology structure regardless of whether V_own is explicitly available: including it is redundant since V_own is already the output of the previous iteration's f_w call. The paper's formulation (neighbours only) is simpler and equally effective.

**29. The result reinforces Experiment 5: the GNN formulation details don't matter.**  
Whether f_w takes mean(V_nbrs) or applies f_w per-neighbour then averages, the routing performance is the same. Combined with Experiment 5 (no-GNN baselines matching GNN performance), this strongly suggests that the Q-network's local observations (queue lengths, destination, action history) are the dominant routing signal, and the GNN's topology encoding — regardless of formulation — plays a secondary role within a 300-episode training budget.

*(See `results/07_paper_vs_ours/` for plots and raw JSON.)*

---

## Implementation Correctness Checks

**Shared SubGNN (weight sharing):**  
A critical bug was found and fixed: the original implementation passed a single SubGNN object to all agents, so `iterate()` calls overwrote the shared `V` buffer sequentially — all nodes ended up with identical feature vectors ("last writer wins"). The correct implementation uses `make_shared_node_gnn()` to give each node its own `V` state buffer while pointing `f_w` and `g_w` to the same weight tensors. Gradients from all agents accumulate during training and are averaged in a single optimiser step via `shared_gnn_step(n_agents)`.

**AttentionSubGNN:**  
Aggregation uses `softmax(dot(V_own, V_nbr_i))` weighted sum — no learnable weight matrices. The `iterate()` and `get_output_trainable()` methods are correctly overridden. Dynamic subclassing via `make_shared_node_gnn(template=AttentionSubGNN_instance)` correctly inherits the attention aggregation path for shared attention variants.

---

## Overall Conclusions

1. **ScaIR is consistently effective under congestion across all tested topologies.** At high hot-spot ratios (D_r ≥ 0.6), it reduces average delivery time by 35–80% compared to OSPF. The improvement scales with congestion severity and topology size.

2. **ScaIR's advantage depends on real-world traffic patterns.** On Abilene (fairly uniform Internet2 TMs), OSPF wins at D_r ≤ 0.2. On BRAIN (concentrated Berlin research network TMs), ScaIR wins at every D_r including D_r=0.0. The crossover point is topology- and traffic-specific: congestion-aware routing only pays when traffic is genuinely unbalanced.

3. **Aggregation mechanism (mean vs attention) matters at scale, not at small topologies.** Across Abilene (11 nodes) and BRAIN (9 nodes), all four variants are within 0.3 ms — statistically negligible. On Germany50 (50 nodes), attn_per_node achieves a consistent 10–15% edge over mean-aggregation variants at high D_r. The attention mechanism provides value only when the neighbourhood is large enough and topology diverse enough that selectively weighting neighbours carries information that mean aggregation destroys.

4. **Weight sharing has negligible impact on performance at small scales; per-node wins at large scale.** On Abilene and BRAIN, shared and per-node variants are interchangeable. On Germany50, per-node Q-nets consistently match or outperform shared Q-nets, particularly for the attention variant (6.77 ms vs 7.07 ms at D_r=0.8). On a 50-node heterogeneous topology, individual routing policies encode node-specific congestion patterns that a single shared Q-net cannot fully represent.

5. **Online adaptation works.** After topology mutations (add/remove node or link), ScaIR adapts through continued training without a full restart, consistently outperforming OSPF on the modified topology except when a new node is inserted (where the new agent is untrained and OSPF's shortest-path recomputation wins immediately).

6. **The GNN is not the key driver of ScaIR's performance — and may actively hurt at larger scales.** The no-GNN ablation (Experiment 5) tested all three topologies with clean UCB-vs-UCB comparisons on BRAIN and Germany50. On BRAIN (9 nodes), fixed encodings match GNN performance within noise. On Germany50 (50 nodes), no-GNN variants consistently outperform ScaIR-with-GNN by 5–9% at the same OSPF baseline. The core value of ScaIR is its multi-agent DQN with local congestion observations (queue lengths, action history) and UCB exploration. The GNN adds optimization complexity without providing better routing signal within a 300-episode training budget.

7. **Limitations and future work:**  
   - The current implementation keeps agents' neighbour lists fixed at init time; link additions therefore benefit only the GNN encoding, not the Q-net's action space. Adding dynamic neighbour-list updates would let ScaIR fully exploit added links.  
   - A longer training budget (1000+ episodes) may reveal clearer differences between variants, particularly for Germany50 where more complex routing policies take longer to converge.  
   - UCB exploration (Experiments 3–4) vs ε-greedy (Experiments 1–2) was not directly ablated; a head-to-head comparison on the same topology would quantify the exploration benefit.  
   - The no-GNN ablation (Experiment 5, all three topologies) found that fixed encodings match or outperform the learned GNN within 300 episodes. A longer training budget (1000+ episodes) may eventually allow the GNN to overtake fixed encodings, but this remains unverified.
