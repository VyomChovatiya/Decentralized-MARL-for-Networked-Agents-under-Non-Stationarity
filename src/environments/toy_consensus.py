"""
Decentralized Networked Binary Consensus Environment (Track 1)
Compatible with Gymnasium / PettingZoo parallel environment patterns.
"""
from typing import Dict, Any, Tuple, List, Optional, Set
import numpy as np
import networkx as nx
from .graphs import generate_topology


class NetworkedConsensusEnv:
    """
    Decentralized Binary Consensus over an arbitrary graph topology.

    At each episode, a target consensus state (0 or 1) is partially observed across
    the network: each agent only receives a noisy local measurement of the true state
    and its own current vote.
    - Without communication (0-hop IPPO), agents cannot aggregate neighbor sensors,
      making unanimous consensus slow and error-prone.
    - With k-hop GNN communication (Networked-PPO / BayesG), agents pool neighborhood
      signals to cancel local noise and rapidly reach unanimous agreement.
    """

    def __init__(
        self,
        num_agents: int = 10,
        topology_type: str = "ring",
        max_steps: int = 30,
        graph: Optional[nx.Graph] = None,
        sensor_noise_std: float = 1.20,
        seed: int = 42,
        **graph_kwargs,
    ):
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.sensor_noise_std = sensor_noise_std
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if graph is not None:
            self.graph = graph
            self.num_agents = graph.number_of_nodes()
        else:
            self.graph = generate_topology(topology_type, num_agents, seed=seed, **graph_kwargs)

        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.current_step = 0
        self.votes = np.zeros(self.num_agents, dtype=np.int64)
        self.local_signals = np.zeros(self.num_agents, dtype=np.float32)
        self.target_consensus: int = 1

        self.neighbors: List[List[int]] = [
            list(self.graph.neighbors(i)) for i in range(self.num_agents)
        ]

        # Observation dimension: [own_vote_signed, static_local_sensor, normalized_degree]
        self.obs_dim = 3
        self.act_dim = 2  # 0: vote 0, 1: vote 1

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.target_consensus = int(self.rng.integers(0, 2))
        target_signed = 2.0 * float(self.target_consensus) - 1.0

        # Fixed per-episode noisy local sensor reading for each agent:
        # Individual agent SNR is low (~30% wrong sign), but graph neighborhood average has high SNR
        self.local_signals = (
            target_signed + self.rng.normal(0.0, self.sensor_noise_std, size=self.num_agents)
        ).astype(np.float32)

        # Initialize votes based on local noisy sensor sign
        self.votes = (self.local_signals >= 0.0).astype(np.int64)
        if np.all(self.votes == self.target_consensus):
            self.votes[0] = 1 - self.target_consensus

        obs = self._get_observations()
        info = {
            "consensus": self.is_consensus(),
            "current_step": self.current_step,
            "target_consensus": self.target_consensus,
        }
        return obs, info

    def _get_observations(self) -> Dict[str, np.ndarray]:
        obs = {}
        for i, agent_id in enumerate(self.agents):
            own_vote_signed = 2.0 * float(self.votes[i]) - 1.0
            deg = len(self.neighbors[i]) / max(1, self.num_agents - 1)
            obs[agent_id] = np.array(
                [own_vote_signed, float(self.local_signals[i]), deg], dtype=np.float32
            )
        return obs

    def is_consensus(self, active_indices: Optional[List[int]] = None) -> bool:
        if active_indices is not None and len(active_indices) > 0:
            sub_votes = self.votes[active_indices]
            return bool(np.all(sub_votes == self.target_consensus))
        return bool(np.all(self.votes == self.target_consensus))

    def step(
        self,
        actions: Dict[str, int],
        shocked_agents: Optional[Set[str]] = None,
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        self.current_step += 1

        for i, agent_id in enumerate(self.agents):
            act = int(actions.get(agent_id, 0))
            self.votes[i] = 1 if act == 1 else 0

        if shocked_agents:
            active_indices = [
                i for i, a in enumerate(self.agents) if a not in shocked_agents
            ]
        else:
            active_indices = list(range(self.num_agents))

        if len(active_indices) == 0:
            active_indices = list(range(self.num_agents))

        consensus_reached = self.is_consensus(active_indices)
        truncated = self.current_step >= self.max_steps
        terminated = consensus_reached

        active_votes = self.votes[active_indices]
        agreement_ratio = float(np.mean(active_votes == self.target_consensus))
        vote_mean = float(np.mean(active_votes))
        disagreement_penalty = -2.5 * vote_mean * (1.0 - vote_mean)
        alignment_shaping = 2.0 * (agreement_ratio - 0.5)
        consensus_bonus = 12.0 if consensus_reached else -0.4

        global_step_reward = disagreement_penalty + alignment_shaping + consensus_bonus

        rewards = {}
        for i, agent_id in enumerate(self.agents):
            indiv_bonus = 0.6 if self.votes[i] == self.target_consensus else -0.6
            rewards[agent_id] = float(global_step_reward + indiv_bonus)

        terminations = {agent_id: terminated for agent_id in self.agents}
        truncations = {agent_id: truncated for agent_id in self.agents}

        obs = self._get_observations()
        info = {
            "consensus": consensus_reached,
            "disagreement": abs(disagreement_penalty),
            "agreement_ratio": agreement_ratio,
            "step": self.current_step,
            "mean_vote": vote_mean,
        }

        return obs, rewards, terminations, truncations, info
