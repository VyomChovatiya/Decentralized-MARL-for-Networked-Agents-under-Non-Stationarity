"""
Decentralized Networked Binary Consensus Environment (Track 1)
Compatible with Gymnasium / PettingZoo parallel environment patterns.
"""
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import networkx as nx
from .graphs import generate_topology


class NetworkedConsensusEnv:
    """
    Decentralized Binary Consensus over an arbitrary graph topology.

    Agents try to reach unanimous agreement (all 0 or all 1) in minimal steps.
    Each agent only observes its own state and its 1-hop graph neighbors' states.
    """

    def __init__(
        self,
        num_agents: int = 10,
        topology_type: str = "ring",
        max_steps: int = 50,
        graph: Optional[nx.Graph] = None,
        seed: int = 42,
        **graph_kwargs,
    ):
        self.num_agents = num_agents
        self.max_steps = max_steps
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

        # Precompute neighbor lists for fast step execution
        self.neighbors: List[List[int]] = [
            list(self.graph.neighbors(i)) for i in range(self.num_agents)
        ]

        # Observation dimension: [own_vote, neighbor_vote_ratio, neighbor_count]
        # or local vector representation
        self.obs_dim = 3
        self.act_dim = 2  # 0: keep, 1: flip

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        # Initialize random initial votes (approx 50/50 split to make consensus non-trivial)
        self.votes = self.rng.integers(0, 2, size=self.num_agents, dtype=np.int64)
        # Avoid starting in already consensus state
        if np.all(self.votes == self.votes[0]):
            self.votes[0] = 1 - self.votes[0]

        obs = self._get_observations()
        info = {"consensus": self.is_consensus(), "current_step": self.current_step}
        return obs, info

    def _get_observations(self) -> Dict[str, np.ndarray]:
        obs = {}
        for i, agent_id in enumerate(self.agents):
            nbrs = self.neighbors[i]
            if len(nbrs) > 0:
                nbr_votes = self.votes[nbrs]
                nbr_ratio = float(np.mean(nbr_votes))
            else:
                nbr_ratio = float(self.votes[i])

            # Local observation vector: [own_vote, neighbor_ratio_of_1s, normalized_degree]
            deg = len(nbrs) / max(1, self.num_agents - 1)
            obs[agent_id] = np.array(
                [float(self.votes[i]), nbr_ratio, deg], dtype=np.float32
            )
        return obs

    def is_consensus(self) -> bool:
        return bool(np.all(self.votes == self.votes[0]))

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        self.current_step += 1

        # Apply actions: 0 = keep, 1 = flip vote
        for i, agent_id in enumerate(self.agents):
            act = actions.get(agent_id, 0)
            if act == 1:
                self.votes[i] = 1 - self.votes[i]

        consensus_reached = self.is_consensus()
        truncated = self.current_step >= self.max_steps
        terminated = consensus_reached

        # Reward formulation:
        # Penalize disagreement (variance of votes across network)
        # Bonus for achieving consensus
        vote_mean = float(np.mean(self.votes))
        disagreement_penalty = -4.0 * vote_mean * (1.0 - vote_mean)  # in [-1, 0]
        consensus_bonus = 10.0 if consensus_reached else 0.0

        step_reward = disagreement_penalty + consensus_bonus

        rewards = {agent_id: step_reward for agent_id in self.agents}
        terminations = {agent_id: terminated for agent_id in self.agents}
        truncations = {agent_id: truncated for agent_id in self.agents}

        obs = self._get_observations()
        info = {
            "consensus": consensus_reached,
            "disagreement": abs(disagreement_penalty),
            "step": self.current_step,
            "mean_vote": vote_mean,
        }

        return obs, rewards, terminations, truncations, info
