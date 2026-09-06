"""
Non-Stationarity Shock Controller (Track 3)
Introduces mid-training policy shifts: t < t_c: pi_A, t >= t_c: pi_B.
"""
from typing import List, Dict, Optional
import numpy as np
import networkx as nx


class NonStationarityShockController:
    """
    Manages mid-training policy perturbation on a subset of agents.
    """

    def __init__(
        self,
        shock_step: int = 500,
        fraction: float = 0.3,
        selection_mode: str = "random",
        behavior: str = "adversarial",
        graph: Optional[nx.Graph] = None,
        num_agents: int = 10,
        seed: int = 42,
    ):
        self.shock_step = shock_step
        self.fraction = fraction
        self.selection_mode = selection_mode
        self.behavior = behavior
        self.num_agents = num_agents
        self.rng = np.random.default_rng(seed)

        # Select target nodes
        k = max(1, int(num_agents * fraction))
        if selection_mode == "hubs" and graph is not None:
            # Nodes sorted by descending degree
            degree_order = sorted(graph.degree, key=lambda x: x[1], reverse=True)
            self.shocked_nodes = [node for node, _ in degree_order[:k]]
        elif selection_mode == "boundary" and graph is not None:
            # Nodes sorted by ascending degree
            degree_order = sorted(graph.degree, key=lambda x: x[1])
            self.shocked_nodes = [node for node, _ in degree_order[:k]]
        else:
            # Uniform random selection
            self.shocked_nodes = list(self.rng.choice(num_agents, size=k, replace=False))

        self.shocked_agents = {f"agent_{i}" for i in self.shocked_nodes}
        self.is_active = False

    def check_and_update(self, current_step: int) -> bool:
        """Returns True if shock is active at current_step"""
        self.is_active = current_step >= self.shock_step
        return self.is_active

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
                # Invert decision to disrupt consensus
                orig_act = actions.get(agent_id, 0)
                modified_actions[agent_id] = 1 - orig_act
            elif self.behavior == "stubborn_0":
                modified_actions[agent_id] = 0
            elif self.behavior == "stubborn_1":
                modified_actions[agent_id] = 1
        return modified_actions
