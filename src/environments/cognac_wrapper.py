"""
COGNAC Benchmark Environment Wrapper (Track 1)
Wraps official COGNAC environments or falls back to native NetworkedConsensusEnv.
"""
from typing import Dict, Any, Tuple, Optional
import networkx as nx
from .graphs import generate_topology
from .toy_consensus import NetworkedConsensusEnv


def make_networked_env(
    env_name: str = "consensus",
    topology_type: str = "ring",
    num_agents: int = 10,
    max_steps: int = 50,
    seed: int = 42,
    **kwargs,
):
    """
    Factory function creating a networked MARL environment.

    Args:
        env_name: 'consensus' or 'cognac_sysadmin'
        topology_type: 'ring', 'grid', 'erdos_renyi', 'scale_free'
        num_agents: number of networked agents
        max_steps: episode horizon
        seed: random seed
    """
    G = generate_topology(topology_type, num_agents, seed=seed, **kwargs)

    # In future weeks, if official cognac package is installed:
    # try:
    #     import cognac
    #     ...
    # except ImportError:
    #     pass

    # Default to high-performance native NetworkedConsensusEnv
    return NetworkedConsensusEnv(
        num_agents=num_agents,
        topology_type=topology_type,
        max_steps=max_steps,
        graph=G,
        seed=seed,
        **kwargs,
    )
