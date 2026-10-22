# Decentralized MARL for Networked Agents under Non-Stationarity

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Benchmark: COGNAC](https://img.shields.io/badge/Benchmark-COGNAC%20(NeurIPS%202025)-green.svg)](https://github.com/yojul/cognac-benchmark-example)
[![Algorithm: BayesG](https://img.shields.io/badge/Algorithm-BayesG%20(NeurIPS%202025)-purple.svg)](https://arxiv.org/abs/2509.16606)

> **RL Research Project** studying how decentralized reinforcement learning agents situated on networked topologies coordinate when neighboring agents update or switch policies concurrently during training.

---

## 📌 Problem Formulation & Motivation

In decentralized Multi-Agent Reinforcement Learning (MARL), each agent $i \in \{1, \dots, N\}$ observes only local state $o_i = O_i(s)$ and interacts with a physical graph topology $G = (V, E)$. The environment transition dynamics from agent $i$'s perspective:

$$P(s_{t+1} \mid s_t, a_i^t) = \sum_{a_{-i}} P(s_{t+1} \mid s_t, a_i^t, a_{-i}^t) \prod_{j \neq i} \pi_j(a_j^t \mid o_j^t)$$

Because neighboring agents simultaneously update their policies ($\pi_j^t \to \pi_j^{t+1}$), the transition kernel continuously changes over time, breaking the stationarity assumption of the Markov Decision Process (MDP).

In networked systems (traffic lights, smart power grids, sensor swarms):
1. **Locality of Observation**: Agents cannot see global state or global rewards.
2. **Constrained Communication**: Agents can only exchange messages with $k$-hop physical neighbors.
3. **Behavioral Shocks**: If a cluster of agents changes strategy mid-training ($t < t_c: \pi_A, t \ge t_c: \pi_B$), shockwaves propagate across the graph.

### Research Questions
1. How does the underlying **network topology** (Ring vs. 2D Grid vs. Erdős-Rényi vs. Barabási-Albert) affect the speed and stability of policy convergence under non-stationarity?
2. How does communication radius $k$ affect adaptation speed when agents switch behaviors?
3. Can **Bayesian Ego-Graph Inference (BayesG)** adaptively prune corrupted or non-stationary neighbors more effectively than static GNN communication?

---

## 👥 3-Person Team Architecture ("Least-Hassle" Collaboration)

To ensure zero merge conflicts and seamless parallel progress, responsibilities are cleanly divided into three decoupled tracks:

| Team Member | Role & Track | Primary Ownership | Deliverables |
|---|---|---|---|
| **Member 1 (Lead)** | **Environments & Topologies** | `src/environments/` | NetworkX graph generators (Ring, Grid, ER, BA), COGNAC PettingZoo wrapper, Non-stationarity policy shock orchestrator ($t_c: \pi_A \to \pi_B$). |
| **Member 2** | **Algorithms & Architectures** | `src/models/`, `src/algorithms/` | Decentralized Actor-Critic, $k$-hop GNN message passing layers, BayesG latent mask variational inference (ELBO objective). |
| **Member 3** | **Experiments & Evaluation** | `src/experiments/`, `src/evaluation/`, `configs/` | YAML config engine, CLI training runner, 6-metric evaluation suite (Instability Index, Adaptation Latency, etc.), automated plotting pipeline. |

---

## 📁 Repository Structure

```
├── configs/                  # Experiment configuration files (YAML)
│   ├── default.yaml          # Base hyperparameters
│   ├── smoke_test.yaml       # Quick 5-node verification test
│   └── scaling_sweep.yaml    # Sweeps across N in {5, 10, 25, 50, 100}
├── papers/                   # Foundational research papers
│   ├── BayesG_NeurIPS_2025.pdf
│   ├── COGNAC_NeurIPS_2025.pdf
│   ├── Adaptive_Partner_Modeling_2024.pdf
│   ├── Distributed_Entropy_Policy_Consensus_2024.pdf
│   └── Open_Ad_Hoc_Teamwork_CIAO_ICML_2024.pdf
├── notebooks/                # Interactive tutorials for team learning
│   └── 01_team_quickstart.ipynb
├── src/
│   ├── environments/         # Track 1: Graph generation & COGNAC wrappers
│   │   ├── graphs.py         # Ring, Grid, ER, Scale-Free generators
│   │   ├── cognac_wrapper.py # PettingZoo COGNAC environment wrapper
│   │   └── toy_consensus.py  # Lightweight standalone consensus environment
│   ├── models/               # Track 2: Neural network architectures
│   │   ├── actors.py         # Decentralized policy networks
│   │   ├── critics.py        # Value function networks
│   │   ├── gnn_comm.py       # k-hop GNN message passing
│   │   └── bayes_ego.py      # BayesG latent ego-graph variational inference
│   ├── algorithms/           # Track 2: MARL algorithms
│   │   ├── ippo.py           # Independent PPO baseline
│   │   ├── networked_ppo.py  # Static GNN communication PPO
│   │   └── bayesg.py         # BayesG algorithm with ELBO loss
│   ├── experiments/          # Track 3: Experiment runner & shock protocol
│   │   ├── shock_protocol.py # Policy switch (t_c) controller
│   │   └── run_experiment.py # YAML-driven experiment entrypoint
│   └── evaluation/           # Track 3: Metrics & plotting
│       ├── metrics.py        # Instability index, adaptation speed, comm cost
│       └── plot_results.py   # Publication-ready plots (learning curves, radars)
├── tests/                    # Unit tests for quick verification
│   ├── test_graphs.py
│   └── test_models.py
├── Project Topic             # Original course topic specification
├── ROADMAP.md                # 8-week team curriculum & weekly milestones
├── pyproject.toml
└── requirements.txt
```

---

## ⚡ Quickstart & Installation

### Option A: Using `uv` (Recommended — Fast & Isolated)
```bash
# 1. Install uv (standalone userspace installer, no root required)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create Python 3.11 environment and install dependencies
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e .
```

### Option B: Using standard `pip`
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## 🚀 Running Experiments

All experiments are 100% **config-driven**. Team members never need to edit code to test new scenarios:

### 1. Run a 5-node Smoke Test
```bash
python src/experiments/run_experiment.py --config configs/smoke_test.yaml
```

### 2. Run Topology Comparison (Ring vs Grid vs ER vs Scale-Free)
```bash
python src/experiments/run_experiment.py --config configs/default.yaml --topology ring --num_agents 10
python src/experiments/run_experiment.py --config configs/default.yaml --topology scale_free --num_agents 10
```

### 3. Generate Comparative Plots & Metrics
```bash
python src/evaluation/plot_results.py --logdir runs/
```

---

## 📊 Core Evaluation Metrics

1. **Convergence Speed**: Number of environment steps to reach 90% asymptotic return.
2. **Mean Return**: Cooperative episodic reward averaged across all agents.
3. **Instability Index**: Rolling variance of policy returns $\text{Var}(R_{[t-\Delta, t+\Delta]})$ around the shock point $t_c$.
4. **Adaptation Latency**: Steps required to recover $\ge 90\%$ of pre-shock return following policy change at $t_c$.
5. **Communication Overhead**: Active communication edges used per step (normalized by full graph density).
6. **Scalability**: Wall-clock time and VRAM footprint scaling across $N \in \{5, 10, 25, 50, 100\}$.

---

## 📚 References & Background Literature

- **BayesG**: Wei Duan, Jie Lu, Junyu Xuan. *Bayesian Ego-Graph Inference for Networked Multi-Agent Reinforcement Learning*. NeurIPS 2025.
- **COGNAC**: Jules Sintes, Ana Bušić. *COGNAC: Cooperative Graph-Based Networked Agent Challenges for MARL*. NeurIPS 2025 (Datasets & Benchmarks).
- **DAPM**: Chenhang Xu et al. *Decentralized Multi-Agent Cooperation via Adaptive Partner Modeling*. Complex & Intelligent Systems, 2024.
- **Policy Consensus**: Yifan Hu et al. *Distributed Entropy-Regularized Multi-Agent Reinforcement Learning with Policy Consensus*. Automatica, 2024.
- **CIAO**: *Open Ad Hoc Teamwork with Cooperative Game Theory*. ICML 2024.

---

## 🔬 Experimental Suite & Empirical Results (Priorities 1–4)

### 1. Unified Training & Hardware Support (Priority 1)
All three decentralized MARL algorithms are fully implemented with multi-agent Generalized Advantage Estimation (GAE) and unified under [`src/experiments/run_experiment.py`](src/experiments/run_experiment.py):
- **`IPPOTrainer` (`ippo`)**: 0-hop decentralized Actor-Critic baseline without inter-agent communication.
- **`NetworkedPPOTrainer` (`networked_ppo`)**: Static $k$-hop GNN message passing with local skip-projections and joint Actor-Critic-GNN optimization.
- **`BayesGTrainer` (`bayesg`)**: Dynamic Bayesian Ego-Graph variational inference (`BayesGEgoInference`) using Gumbel-Softmax edge masks $Z_{ij} \in [0, 1]$, pairwise relational difference features $[h_i, h_j, |h_i - h_j|]$, and joint PPO + ELBO KL optimization.
- **Hardware Acceleration**: Auto-detects `cuda`, Intel Arc `xpu`, or `cpu`.

#### Running the Full Priority 1–4 Benchmark Suite
```bash
# Run all 19 unit tests (Environments, Topologies, Models, and Trainer Updates)
pytest tests/

# Generate 4-topology visual comparison (Ring, Grid, Erdos-Renyi, Scale-Free)
python src/evaluation/plot_topologies.py

# Run the complete Priority 2-4 experimental suite & generate all figures
python src/experiments/run_priority_suite.py
```

---

### 2. Normal Operation Baseline — No Shocks (Priority 2)
Evaluated on a 10-node Ring topology (`N=10, k=2, 90 episodes, shock_step=None`) where each agent receives a noisy local sensor measurement of the target consensus state:

| Algorithm | Mean Return ($\uparrow$) | Max Smoothed Return ($\uparrow$) | Convergence Episode ($\downarrow$) | Avg Comm Cost |
| :--- | :---: | :---: | :---: | :---: |
| **IPPO (0-Hop, No Comm)** | `-5.61` | `9.93` | `41` | `0.00` |
| **Networked-PPO ($k$-Hop GNN)** | `+1.10` | `13.14` | `65` | `1.00` |
| **BayesG (Dynamic Ego-Graph)** | **`+5.57`** | **`14.78`** | **`36`** | `0.90` |

* **Finding**: Multi-hop communication (`Networked-PPO` and `BayesG`) substantially outperforms isolated learning (`IPPO`) by pooling neighborhood observations to cancel local sensor noise.

---

### 3. Non-Stationarity Shock Benchmarks (Priority 3)

#### 3A. Mid-Training Random Policy Shock (`Ring, N=10, t_c=50, fraction=0.3`)
At episode $t_c = 50$, 30% of agents switch to an adversarial policy and broadcast shifted/corrupted state features:

| Algorithm | Mean Return ($\uparrow$) | Instability Index ($\downarrow$) | Adaptation Latency ($\downarrow$) | Avg Comm Cost ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: |
| **IPPO (0-Hop)** | `-3.52` | `19.42` | `3 eps` | `0.00` |
| **Networked-PPO (Static GNN)** | `-4.83` | `25.69` | `22 eps` | `1.00` |
| **BayesG (Dynamic Mask)** | **`+5.08`** | **`8.89`** | **`3 eps`** | **`0.75`** |

* **Finding**: Static `Networked-PPO` suffers severe instability (`25.69`) and slow recovery (`22 episodes`) because corrupted neighbor features propagate across the graph. `BayesG` reduces instability by **~3x (`8.89`)** and recovers in **3 episodes** while cutting active communication overhead to `0.75`.

#### 3B. Scale-Free (Barabási-Albert) Network: Hub Shock vs. Periphery Shock (`N=12, t_c=50`)
Compares shocking high-degree **hub nodes** (`[0, 1, 4]`) versus low-degree **boundary/periphery nodes** (`[2, 7, 9]`):

| Scenario | Shocked Nodes | Mean Return ($\uparrow$) | Instability Index ($\downarrow$) | Shock Drop ($\downarrow$) | Adaptation Latency ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NetPPO (Hub Shock)** | `[0, 1, 4]` | `-11.74` | `30.74` | `60.71` | `20 eps` |
| **NetPPO (Boundary Shock)** | `[2, 7, 9]` | `-6.75` | `23.44` | `61.98` | `4 eps` |
| **BayesG (Hub Shock)** | `[0, 1, 4]` | **`+0.33`** | **`15.62`** | **`25.82`** | **`3 eps`** |
| **BayesG (Boundary Shock)** | `[2, 7, 9]` | **`+4.32`** | **`11.99`** | **`17.29`** | **`3 eps`** |

* **Finding**: Scale-free networks are disproportionately vulnerable to hub shocks under static GNN message passing (`20 episodes` recovery latency vs. `4 episodes` for boundary shocks). `BayesG` mitigates hub vulnerability by isolating perturbed hubs, cutting instability and shock drop by **>50%**.

---

### 4. Dynamic Ego-Graph Edge Pruning Visualization (Priority 4)
Quantitative comparison of inferred communication edge probabilities $\bar{Z}_{ij}$ in `BayesG` before ($t < t_c$) and after ($t > t_c$) the Scale-Free Hub Shock:

- **Edges connected to Shocked Hubs `[0, 1, 4]`**:
  - Pre-Shock ($t < t_c$): **`0.904`** $\longrightarrow$ Post-Shock ($t > t_c$): **`0.267`** (**>70% edge weight suppression / topological quarantine**)
- **Edges between Healthy Operational Nodes**:
  - Pre-Shock ($t < t_c$): **`0.910`** $\longrightarrow$ Post-Shock ($t > t_c$): **`0.901`** (**healthy communication links preserved at 90%**)

Generated figures are saved to:
- `results/topologies_overview.png`
- `results/priority2_no_shock/no_shock_comparison.png`
- `results/priority3_random_shock/shock_comparison.png`
- `results/priority3_hub_vs_periphery/hub_vs_periphery.png`
- `results/priority4_edge_pruning/bayesg_edge_pruning.png`
