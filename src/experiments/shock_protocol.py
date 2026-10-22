"""
Non-Stationarity Shock Controller (Track 3)
Introduces mid-training policy shifts: t < t_c: pi_A, t >= t_c: pi_B.
"""
from typing import List, Dict, Optional
import numpy as np
import networkx as nx


class NonStationarityShockController:
    """
    Manages mid-training policy and feature perturbation on a subset of agents.
    Supports targeting random nodes, high-degree hub nodes, or peripheral boundary nodes.
    """

    def __init__(
        self,
        shock_step: Optional[int] = 500,
        fraction: float = 0.3,
        selection_mode: str = "random",
        behavior: str = "adversarial",
        graph: Optional[nx.Graph] = None,
        num_agents: int = 10,
        seed: int = 42,
    ):
        self.shock_step = shock_step
        self.fraction = fraction
        self.selection_mode = selection_mode.lower() if selection_mode else "none"
        self.behavior = behavior
        self.num_agents = num_agents
        self.rng = np.random.default_rng(seed)

        if (
            shock_step is None
            or shock_step < 0
            or fraction <= 0.0
            or self.selection_mode == "none"
        ):
            self.shocked_nodes: List[int] = []
        else:
            k = max(1, int(round(num_agents * fraction)))
            if self.selection_mode == "hubs" and graph is not None:
                degree_order = sorted(graph.degree, key=lambda x: x[1], reverse=True)
                self.shocked_nodes = [int(node) for node, _ in degree_order[:k]]
            elif self.selection_mode in ["boundary", "periphery", "peripheral"] and graph is not None:
                degree_order = sorted(graph.degree, key=lambda x: x[1])
                self.shocked_nodes = [int(node) for node, _ in degree_order[:k]]
            else:
                self.shocked_nodes = [
                    int(x) for x in self.rng.choice(num_agents, size=k, replace=False)
                ]

        self.shocked_agents = {f"agent_{i}" for i in self.shocked_nodes}
        self.is_active = False

    def check_and_update(self, current_step: int) -> bool:
        """Returns True if shock is active at current_step"""
        if self.shock_step is None or self.shock_step < 0 or len(self.shocked_nodes) == 0:
            self.is_active = False
            return False
        self.is_active = current_step >= self.shock_step
        return self.is_active

    def apply_shock_observations(
        self, current_step: int, obs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Applies non-stationary policy/sensor feature shift to shocked agents'
        broadcasted observations when current_step >= shock_step.
        """
        if not self.check_and_update(current_step):
            return obs

        modified_obs = {k: v.copy() for k, v in obs.items()}
        for agent_id in self.shocked_agents:
            if agent_id not in modified_obs:
                continue
            vec = modified_obs[agent_id]
            if self.behavior == "adversarial":
                # Invert state and inject strong adversarial bias into transmitted message
                vec[0] = -vec[0] * 3.5
                vec[1] = -vec[1] * 4.0 - 3.5
            elif self.behavior == "random":
                vec[0] = float(self.rng.choice([-3.0, 3.0]))
                vec[1] = float(self.rng.normal(0.0, 4.5))
            elif self.behavior == "stubborn_0":
                vec[0] = -3.5
                vec[1] = -4.0
            elif self.behavior == "stubborn_1":
                vec[0] = 3.5
                vec[1] = 4.0
            modified_obs[agent_id] = vec
        return modified_obs

    def apply_shock_actions(
        self, current_step: int, actions: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Overrides actions of shocked agents if step >= shock_step.
        """
        if not self.check_and_update(current_step):
            return actions

        modified_actions = dict(actions)
        for agent_id in self.shocked_agents:
            if self.behavior == "random":
                modified_actions[agent_id] = int(self.rng.integers(0, 2))
            elif self.behavior == "adversarial":
                orig_act = actions.get(agent_id, 0)
                modified_actions[agent_id] = 1 - orig_act
            elif self.behavior == "stubborn_0":
                modified_actions[agent_id] = 0
            elif self.behavior == "stubborn_1":
                modified_actions[agent_id] = 1
        return modified_actions
