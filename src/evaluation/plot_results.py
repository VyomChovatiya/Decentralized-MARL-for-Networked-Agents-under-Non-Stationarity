"""
Automated Plotting and Visualization Pipeline (Track 3)
Generates publication-quality learning curves, shock recovery plots, and topology comparisons.
"""
from typing import List, Dict, Optional
import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


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
        kernel = np.ones(window) / window
        smoothed = np.convolve(returns, kernel, mode="valid")
        smooth_episodes = np.arange(window, len(returns) + 1)
        ax.plot(smooth_episodes, smoothed, color="royalblue", linewidth=2.2, label=f"Moving Avg ({window} eps)")

    if shock_step is not None:
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
