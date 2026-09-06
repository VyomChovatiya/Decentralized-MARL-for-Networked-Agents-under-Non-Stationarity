# Decentralized MARL for Networked Agents under Non-Stationarity

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark: COGNAC](https://img.shields.io/badge/Benchmark-COGNAC%20(NeurIPS%202025)-green.svg)](papers/COGNAC_NeurIPS_2025.pdf)
[![Algorithm: BayesG](https://img.shields.io/badge/Algorithm-BayesG%20(NeurIPS%202025)-purple.svg)](papers/BayesG_NeurIPS_2025.pdf)

> **2-Month Undergraduate Research Project** studying how decentralized reinforcement learning agents situated on networked topologies coordinate when neighboring agents update or switch policies concurrently during training.

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
| **Member 1 (Vyom - Lead)** | **Environments & Topologies** | `src/environments/` | NetworkX graph generators (Ring, Grid, ER, BA), COGNAC PettingZoo wrapper, Non-stationarity policy shock orchestrator ($t_c: \pi_A \to \pi_B$). |
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
