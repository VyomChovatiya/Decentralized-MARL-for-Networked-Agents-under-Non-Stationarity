"""
Tests for Graph Topology Generators (Track 1)
"""
import pytest
import networkx as nx
from src.environments.graphs import generate_topology, get_adjacency_matrix, compute_graph_metrics


@pytest.mark.parametrize("topology", ["ring", "grid", "erdos_renyi", "scale_free"])
@pytest.mark.parametrize("num_nodes", [5, 10, 20])
def test_graph_generation(topology, num_nodes):
    G = generate_topology(topology, num_nodes, seed=123)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == num_nodes
    assert nx.is_connected(G)


def test_adjacency_normalization():
    G = generate_topology("ring", 6)
    A_norm = get_adjacency_matrix(G, normalized=True)
    assert A_norm.shape == (6, 6)
    # Check symmetric
    import numpy as np
    assert np.allclose(A_norm, A_norm.T)


def test_metrics_computation():
    G = generate_topology("ring", 10)
    metrics = compute_graph_metrics(G)
    assert metrics["num_nodes"] == 10
    assert metrics["diameter"] == 5
    assert metrics["algebraic_connectivity"] > 0
