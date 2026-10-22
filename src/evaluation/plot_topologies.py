"""
Script to visualize the 4 network topologies for presentations and reports.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import matplotlib.pyplot as plt
import networkx as nx
from src.environments.graphs import generate_topology, compute_graph_metrics

def generate_topology_overview(save_path="results/topologies_overview.png", num_nodes=16):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=200)
    topologies = ["ring", "grid", "erdos_renyi", "scale_free"]
    titles = ["Ring (1D Cycle)", "2D Grid (Lattice)", "Erdos-Renyi (Random)", "Barabasi-Albert (Scale-Free)"]

    for ax, top, title in zip(axes, topologies, titles):
        G = generate_topology(top, num_nodes=num_nodes, seed=42)
        stats = compute_graph_metrics(G)
        
        if top == "ring":
            pos = nx.circular_layout(G)
        elif top == "grid":
            pos = {n: (n % 4, -(n // 4)) for n in G.nodes()}
        else:
            pos = nx.spring_layout(G, seed=42)
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#2b5c8f", node_size=280)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#888888", alpha=0.7)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="white")
        diam = stats["diameter"]
        fiedler = stats["algebraic_connectivity"]
        ax.set_title(f"{title}\nDiam={diam}, Fiedler={fiedler:.2f}", fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Topologies overview saved to: {save_path}")

if __name__ == "__main__":
    generate_topology_overview()
