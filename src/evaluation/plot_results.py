"""
Automated Plotting and Visualization Pipeline (Track 3)
Generates publication-quality learning curves, multi-algorithm comparisons,
hub-vs-periphery shock studies, and BayesG edge-pruning heatmaps.
"""
from typing import List, Dict, Optional, Any
import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


ALGO_COLORS = {
    "ippo": "#7f7f7f",
    "networked_ppo": "#1f77b4",
    "bayesg": "#2ca02c",
}

ALGO_LABELS = {
    "ippo": "IPPO (0-Hop, No Comm)",
    "networked_ppo": "Networked-PPO (Static k-Hop GNN)",
    "bayesg": "BayesG (Dynamic Ego-Graph Pruning)",
}


def _smooth(arr: np.ndarray, window: int = 8) -> np.ndarray:
    if len(arr) < window or window <= 1:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_learning_curve(
    returns: List[float],
    title: str = "Decentralized MARL Training Curve",
    shock_step: Optional[int] = None,
    save_path: Optional[str] = None,
    window: int = 10,
):
    """
    Plot episodic returns with moving average and optional shock vertical line.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    episodes = np.arange(1, len(returns) + 1)
    ax.plot(episodes, returns, color="royalblue", alpha=0.25, label="Raw Return")

    if len(returns) >= window:
        smoothed = _smooth(np.array(returns, dtype=np.float64), window)
        smooth_episodes = np.arange(window, len(returns) + 1)
        ax.plot(smooth_episodes, smoothed, color="royalblue", linewidth=2.2, label=f"Moving Avg ({window} eps)")

    if shock_step is not None and shock_step > 0:
        ax.axvline(
            x=shock_step,
            color="crimson",
            linestyle="--",
            linewidth=2,
            label=f"Non-Stationarity Shock (t_c = {shock_step})",
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Training Episode", fontsize=11)
    ax.set_ylabel("Mean Episode Return", fontsize=11)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    plt.close()


def plot_multi_algorithm_curves(
    results_by_algo: Dict[str, Dict[str, Any]],
    title: str = "Algorithm Comparison",
    shock_step: Optional[int] = None,
    save_path: Optional[str] = None,
    window: int = 8,
):
    """
    Plot comparative learning curves for IPPO, Networked-PPO, and BayesG on the same figure.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)

    for algo, data in results_by_algo.items():
        returns = np.array(data["returns"], dtype=np.float64)
        color = ALGO_COLORS.get(algo.lower(), None)
        label = ALGO_LABELS.get(algo.lower(), algo.upper())

        episodes = np.arange(1, len(returns) + 1)
        ax.plot(episodes, returns, color=color, alpha=0.18, linewidth=1.0)

        if len(returns) >= window:
            smoothed = _smooth(returns, window)
            smooth_ep = np.arange(window, len(returns) + 1)
            ax.plot(smooth_ep, smoothed, color=color, linewidth=2.6, label=label)
        else:
            ax.plot(episodes, returns, color=color, linewidth=2.6, label=label)

    if shock_step is not None and shock_step > 0:
        ax.axvline(
            x=shock_step,
            color="crimson",
            linestyle="--",
            linewidth=2.2,
            label=f"Non-Stationarity Shock (t_c = {shock_step})",
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Training Episode", fontsize=11)
    ax.set_ylabel("Cooperative Episode Return", fontsize=11)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Multi-algorithm plot saved to: {save_path}")
    plt.close()


def plot_hub_vs_periphery_comparison(
    comparison_data: Dict[str, Dict[str, Any]],
    shock_step: int,
    save_path: Optional[str] = None,
    window: int = 8,
):
    """
    Plot Scale-Free Hub vs. Periphery shock curves and adaptation metrics side-by-side.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=180)

    style_map = {
        "NetPPO (Hub Shock)": ("#d62728", "-", "Networked-PPO (Hub Shock)"),
        "NetPPO (Boundary Shock)": ("#1f77b4", "--", "Networked-PPO (Periphery Shock)"),
        "BayesG (Hub Shock)": ("#2ca02c", "-", "BayesG (Hub Shock - Quarantined)"),
        "BayesG (Boundary Shock)": ("#98df8a", "--", "BayesG (Periphery Shock)"),
    }

    labels = []
    instabilities = []
    drops = []
    bar_colors = []

    for key, data in comparison_data.items():
        returns = np.array(data["returns"], dtype=np.float64)
        color, ls, label = style_map.get(key, (None, "-", key))
        if len(returns) >= window:
            smoothed = _smooth(returns, window)
            smooth_ep = np.arange(window, len(returns) + 1)
            ax1.plot(smooth_ep, smoothed, color=color, linestyle=ls, linewidth=2.5, label=label)
        else:
            ax1.plot(np.arange(1, len(returns) + 1), returns, color=color, linestyle=ls, linewidth=2.5, label=label)

        labels.append(key.replace(" ", "\n"))
        instabilities.append(data.get("instability_index", 0.0))
        drops.append(max(0.0, data.get("shock_drop", 0.0)))
        bar_colors.append(color or "#4C72B0")

    ax1.axvline(x=shock_step, color="crimson", linestyle=":", linewidth=2.0, label=f"Shock (t_c={shock_step})")
    ax1.set_title("Scale-Free Graph: Hub vs. Periphery Shock", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Training Episode")
    ax1.set_ylabel("Cooperative Episode Return")
    ax1.legend(loc="lower right", frameon=True, fontsize=9)

    x = np.arange(len(labels))
    width = 0.36
    ax2.bar(x - width / 2, instabilities, width=width, color="#C44E52", alpha=0.85, edgecolor="black", label="Instability Index (Std)")
    ax2.bar(x + width / 2, drops, width=width, color="#4C72B0", alpha=0.85, edgecolor="black", label="Immediate Shock Drop")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax2.set_title("Vulnerability by Node Role & Algorithm", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Magnitude (Lower = More Resilient)")
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Hub vs Periphery plot saved to: {save_path}")
    plt.close()


def plot_bayesg_edge_pruning(
    adj_matrix: np.ndarray,
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
    shocked_nodes: List[int],
    save_path: Optional[str] = None,
):
    """
    Visualize BayesG latent ego-graph edge weights Z_ij before and after shock t_c.
    Shows both adjacency heatmaps and network graph pruning diagrams.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    N = adj_matrix.shape[0]
    eye = np.eye(N)
    phys_off_diag = (adj_matrix > 0).astype(float) * (1.0 - eye)

    pre_clean = pre_mask * phys_off_diag
    post_clean = post_mask * phys_off_diag

    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), dpi=180)

    # Row 1: Heatmaps of Z_ij
    im1 = axes[0, 0].imshow(pre_clean, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("Pre-Shock Inferred Edge Mask Z_ij (t < t_c)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Sender Node j")
    axes[0, 0].set_ylabel("Receiver Node i")
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im2 = axes[0, 1].imshow(post_clean, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title(
        f"Post-Shock Inferred Edge Mask Z_ij (t > t_c)\nShocked Nodes: {shocked_nodes} (Quarantined)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 1].set_xlabel("Sender Node j")
    axes[0, 1].set_ylabel("Receiver Node i")
    for s_node in shocked_nodes:
        axes[0, 1].axvline(x=s_node, color="crimson", linestyle="--", alpha=0.7, linewidth=1.5)
        axes[0, 1].axhline(y=s_node, color="crimson", linestyle="--", alpha=0.7, linewidth=1.5)
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Row 2: Network Graph Visualization before & after shock
    G = nx.from_numpy_array(phys_off_diag)
    pos = nx.spring_layout(G, seed=42)
    node_colors = ["#d62728" if i in shocked_nodes else "#2b5c8f" for i in range(N)]

    for ax_idx, (mat, subtitle) in enumerate([
        (pre_clean, "Pre-Shock Active Communication Graph (All Links Healthy)"),
        (post_clean, "Post-Shock Quarantined Graph (Links to Red Shocked Nodes Pruned)"),
    ]):
        ax = axes[1, ax_idx]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=420, edgecolors="black")
        nx.draw_networkx_labels(G, pos, ax=ax, font_color="white", font_size=9, font_weight="bold")

        edges = list(G.edges())
        weights = [0.5 * (mat[u, v] + mat[v, u]) for u, v in edges]
        for (u, v), w in zip(edges, weights):
            is_shock_edge = (u in shocked_nodes) or (v in shocked_nodes)
            color = "#d62728" if (is_shock_edge and ax_idx == 1) else "#2ca02c"
            style = "dotted" if w < 0.35 else "solid"
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                ax=ax,
                width=max(0.4, w * 4.0),
                alpha=max(0.18, float(w)),
                edge_color=color,
                style=style,
            )
        ax.set_title(subtitle, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"BayesG edge pruning visualization saved to: {save_path}")
    plt.close()


def plot_topology_comparison(
    topo_metrics: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
):
    """
    Generate grouped bar charts comparing Ring, Grid, ER, and Scale-Free networks.
    """
    topologies = list(topo_metrics.keys())
    returns = [topo_metrics[t].get("mean_return", 0.0) for t in topologies]
    latencies = [topo_metrics[t].get("adaptation_latency", 0.0) for t in topologies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    x = np.arange(len(topologies))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    # 1. Mean Return
    ax1.bar(x, returns, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(topologies, fontweight="bold")
    ax1.set_title("Mean Episodic Return by Topology", fontsize=12)
    ax1.set_ylabel("Return")

    # 2. Adaptation Latency
    ax2.bar(x, latencies, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(topologies, fontweight="bold")
    ax2.set_title("Adaptation Latency (Steps to Recover)", fontsize=12)
    ax2.set_ylabel("Episodes Post-Shock")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Topology comparison saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_file", type=str, default="results/metrics.json")
    parser.add_argument("--output_dir", type=str, default="results/plots")
    args = parser.parse_args()

    if os.path.exists(args.metrics_file):
        with open(args.metrics_file, "r") as f:
            data = json.load(f)
        if "returns" in data:
            plot_learning_curve(
                data["returns"],
                shock_step=data.get("shock_step"),
                save_path=os.path.join(args.output_dir, "learning_curve.png"),
            )
