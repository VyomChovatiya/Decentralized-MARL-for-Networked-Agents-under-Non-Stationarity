"""
Graph Topology Generators and Analysis (Track 1)
Supports Ring, 2D Grid, Erdős-Rényi, and Barabási-Albert topologies.
"""
from typing import Dict, Any, Tuple
import math
import networkx as nx
import numpy as np
import scipy.sparse as sp


def generate_topology(topology_type: str, num_nodes: int, seed: int = 42, **kwargs) -> nx.Graph:
    """
    Generate a connected network topology.

    Args:
        topology_type: One of 'ring', 'grid', 'erdos_renyi', 'scale_free'
        num_nodes: Total number of agents/nodes
        seed: Random seed for stochastic graphs
        **kwargs: Additional graph-specific parameters (e.g. p for ER, m for BA)

    Returns:
        nx.Graph: Connected undirected networkx Graph
    """
    top = topology_type.lower()
    np.random.seed(seed)

    if top == "ring":
        # Cycle graph: each node connected to 2 neighbors
        G = nx.cycle_graph(num_nodes)

    elif top == "grid":
        # 2D Grid graph: closest factors to a square
        side_a = int(math.isqrt(num_nodes))
        while num_nodes % side_a != 0 and side_a > 1:
            side_a -= 1
        side_b = num_nodes // side_a
        G_grid = nx.grid_2d_graph(side_a, side_b)
        # Relabel nodes to integers 0..num_nodes-1
        G = nx.convert_node_labels_to_integers(G_grid)

    elif top in ["erdos_renyi", "er", "random"]:
        # Random graph G(n, p) - ensure connected
        p = kwargs.get("p", max(0.15, 2.0 * math.log(num_nodes) / num_nodes))
        attempts = 0
        while attempts < 100:
            G = nx.erdos_renyi_graph(num_nodes, p, seed=seed + attempts)
            if nx.is_connected(G):
                break
            attempts += 1
        if not nx.is_connected(G):
            # If still disconnected, add minimal edges to bridge components
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                u = list(components[i])[0]
                v = list(components[i + 1])[0]
                G.add_edge(u, v)

    elif top in ["scale_free", "barabasi_albert", "ba"]:
        # Barabási-Albert scale-free graph with hubs
        m = kwargs.get("m", max(1, min(2, num_nodes - 1)))
        G = nx.barabasi_albert_graph(num_nodes, m, seed=seed)

    else:
        raise ValueError(f"Unknown topology type: {topology_type}. Choose from 'ring', 'grid', 'erdos_renyi', 'scale_free'")

    # Ensure relabeled 0..N-1
    G = nx.convert_node_labels_to_integers(G)
    return G


def get_adjacency_matrix(graph: nx.Graph, normalized: bool = True) -> np.ndarray:
    """
    Compute dense adjacency matrix, optionally symmetric normalized A_hat = D^{-1/2} (A + I) D^{-1/2}.
    """
    A = nx.to_numpy_array(graph, dtype=np.float32)
    if not normalized:
        return A

    # Add self loops
    A_tilde = A + np.eye(graph.number_of_nodes(), dtype=np.float32)
    degrees = np.sum(A_tilde, axis=1)
    deg_inv_sqrt = np.zeros_like(degrees, dtype=np.float32)
    positive_mask = degrees > 0
    deg_inv_sqrt[positive_mask] = np.power(degrees[positive_mask], -0.5)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt


def get_ego_graph(graph: nx.Graph, node: int, radius: int = 1) -> nx.Graph:
    """
    Extract the k-hop ego-graph around a specific node.
    """
    return nx.ego_graph(graph, node, radius=radius)


def compute_graph_metrics(graph: nx.Graph) -> Dict[str, Any]:
    """
    Compute graph-theoretic properties useful for analyzing MARL convergence.
    """
    N = graph.number_of_nodes()
    E = graph.number_of_edges()
    is_connected = nx.is_connected(graph)

    # Spectral properties
    L = nx.laplacian_matrix(graph).toarray().astype(float)
    eigenvalues = np.sort(np.linalg.eigvals(L))
    # Fiedler eigenvalue (algebraic connectivity = 2nd smallest eigenvalue)
    algebraic_connectivity = float(np.real(eigenvalues[1])) if N > 1 else 0.0

    # Path metrics
    if is_connected:
        diameter = nx.diameter(graph)
        avg_path_length = nx.average_shortest_path_length(graph)
    else:
        diameter = float("inf")
        avg_path_length = float("inf")

    avg_clustering = nx.average_clustering(graph)
    degrees = [d for _, d in graph.degree()]

    return {
        "num_nodes": N,
        "num_edges": E,
        "is_connected": is_connected,
        "diameter": diameter,
        "avg_path_length": avg_path_length,
        "algebraic_connectivity": algebraic_connectivity,
        "avg_clustering": avg_clustering,
        "degree_mean": float(np.mean(degrees)),
        "degree_std": float(np.std(degrees)),
        "degree_max": int(np.max(degrees)),
    }
