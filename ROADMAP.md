# 2-Month Project Roadmap & Curriculum (Team of 3)

## 🎯 Executive Summary

- **Duration**: 8 Weeks (~2 Months)
- **Team**: 3 Undergraduate Researchers
- **Objective**: Deliver a complete empirical research study on **Decentralized MARL on Networked Topologies under Non-Stationarity**, evaluating the resilience of **IPPO**, **Static $k$-hop GNN**, and **BayesG** against policy-switching shocks across multiple network structures ($N \in \{5, 10, 25, 50, 100\}$).

---

## 🗓️ Weekly Schedule & Role Matrix

```
Week 1-2: Foundations, Environment Setup & 5-Node Baseline
Week 3-4: Graph Topologies & k-Hop Networked Communication
Week 5-6: Policy-Switch Non-Stationarity Shock & BayesG Integration
Week 7:   Large-Scale Systematic Sweeps (N = 5 to 100) & Ablations
Week 8:   Metrics Analysis, Visualizations & Final Research Report
```

---

### Phase 1: Foundations & Quickstart (Weeks 1–2)

#### Week 1: Concept Ramp-up & Environment Setup
- **Reading**:
  - Sutton & Barto (Ch. 13: Policy Gradients)
  - COGNAC Paper (`papers/COGNAC_NeurIPS_2025.pdf`, Sections 1–2)
  - Project Topic file
- **Tasks by Member**:
  - **Member 1 (Vyom - Environments)**: Study PettingZoo API and COGNAC environment structures; write toy consensus environment with NetworkX visualization.
  - **Member 2 (Algorithms)**: Review Actor-Critic fundamentals and PPO clip objective; write decentralized MLP policy network.
  - **Member 3 (Experiments)**: Setup repository, virtual environment, YAML configuration system, and TensorBoard logging.
- **Weekly Checkpoint**: All 3 members run `python src/experiments/run_experiment.py --config configs/smoke_test.yaml` and see an active training loop.

#### Week 2: Baseline IPPO & Metric Plumbing
- **Reading**:
  - *Distributed Entropy-Regularized MARL* (`papers/Distributed_Entropy_Policy_Consensus_2024.pdf`, Sections 1–3)
- **Tasks by Member**:
  - **Member 1**: Implement NetworkX graph generator (`src/environments/graphs.py`) for Ring and 2D Grid with adjacency matrices.
  - **Member 2**: Implement full Independent PPO (IPPO) agent update (`src/algorithms/ippo.py`).
  - **Member 3**: Implement rolling return logger and episodic metric tracker.
- **Weekly Checkpoint**: IPPO trained on a 5-node Ring graph reaching consensus. Plot learning curves on TensorBoard.

---

### Phase 2: Network Topologies & $k$-Hop Communication (Weeks 3–4)

#### Week 3: Topology Suite (Ring, Grid, ER, Scale-Free)
- **Reading**:
  - Graph theory concepts: Diameter, Degree Distribution, Algebraic Connectivity (Fiedler value).
- **Tasks by Member**:
  - **Member 1**: Add Erdős-Rényi ($G(N, p)$) and Barabási-Albert (Scale-Free) generators to `graphs.py`. Implement graph property analyzer (spectral gap, diameter).
  - **Member 2**: Implement GNN message passing layer (`src/models/gnn_comm.py`) allowing agents to aggregate embeddings from $k \in \{1, 2, 3\}$ neighbor hops.
  - **Member 3**: Create automated config generator for topology matrix sweep ($N=10$, across 4 topologies).
- **Weekly Checkpoint**: Verify GNN message passing forward/backward pass on GPU without memory leaks.

#### Week 4: Networked-PPO Benchmark
- **Tasks by Member**:
  - **Member 1**: Ensure COGNAC wrappers support arbitrary graph injection.
  - **Member 2**: Complete Networked-PPO algorithm (`src/algorithms/networked_ppo.py`) combining local policy with $k$-hop neighbor embeddings.
  - **Member 3**: Benchmark IPPO (0-hop) vs. Networked-PPO (1-hop, 2-hop) across all 4 topologies.
- **Weekly Checkpoint**: Generate first comparative bar chart: Reward and convergence speed as a function of network topology and communication radius $k$.

---

### Phase 3: Non-Stationarity & BayesG Inference (Weeks 5–6)

#### Week 5: Mid-Training Policy-Switch Shock Protocol
- **Reading**:
  - *Adaptive Partner Modeling* (`papers/Adaptive_Partner_Modeling_2024.pdf`, Sections 1–3)
  - *CIAO Open Ad Hoc Teamwork* (`papers/Open_Ad_Hoc_Teamwork_CIAO_ICML_2024.pdf`)
- **Tasks by Member**:
  - **Member 1**: Implement shock protocol (`src/experiments/shock_protocol.py`): at timestep $t_c$, randomly or strategically (e.g. hub nodes vs peripheral nodes) switch $\rho \in \{0.2, 0.5, 0.8\}$ of agents from policy $\pi_A$ to $\pi_B$ (stochastic/adversarial/inverted).
  - **Member 2**: Analyze how static GNNs propagate bad information from shocked nodes to healthy nodes.
  - **Member 3**: Implement **Instability Index** and **Adaptation Latency** metrics in `src/evaluation/metrics.py`.
- **Weekly Checkpoint**: Show the "shock cliff" in learning curves when $t \ge t_c$.

#### Week 6: BayesG Ego-Graph Inference Integration
- **Reading**:
  - BayesG Paper (`papers/BayesG_NeurIPS_2025.pdf`, Sections 3–4)
- **Tasks by Member**:
  - **Member 1**: Add ego-graph subnetwork extraction to `graphs.py`.
  - **Member 2**: Implement BayesG latent mask module (`src/models/bayes_ego.py`): Gumbel-Softmax variational inference with ELBO loss.
  - **Member 3**: Instrument logging of dynamic communication edge weights (confirming that BayesG prunes edges to shocked nodes).
- **Weekly Checkpoint**: Demonstrate that BayesG recovers faster after the shock by cutting off connections to disrupted neighbors.

---

### Phase 4: Scaling Sweeps, Report & Presentation (Weeks 7–8)

#### Week 7: Scaling to $N=50$ and $N=100$ Agents
- **Tasks by Member**:
  - **Member 1**: Test scalability limits on large Barabási-Albert networks ($N=50, 100$). Correlate recovery speed with topological node centrality.
  - **Member 2**: Tune BayesG ELBO regularization and temperature schedules for large agent counts.
  - **Member 3**: Execute full batch sweeps across $N \in \{5, 10, 25, 50, 100\}$; log wall-clock time and GPU memory.
- **Weekly Checkpoint**: Complete data collection for all experimental configurations.

#### Week 8: Publication-Quality Plots, Report & Slide Deck
- **Tasks by Member**:
  - **Member 1**: Write report sections: *Introduction, Problem Statement, Network Topologies & Graph Spectral Analysis*.
  - **Member 2**: Write report sections: *Methodology (IPPO, GNN, BayesG Variational Inference)*.
  - **Member 3**: Generate final figures (learning curves with shaded std error, radar plots, heatmaps) and write *Experiments, Results, and Discussion*.
- **Final Deliverables Checklist**:
  - [ ] GitHub Repository with clean code, documentation, and reproducing scripts.
  - [ ] 8–10 page Research Paper / Final Project Report in standard NeurIPS/IEEE LaTeX template.
  - [ ] Slide Deck (15–20 slides) for project presentation.
  - [ ] Interactive demo or video showing agents coordinating and overcoming non-stationarity.
