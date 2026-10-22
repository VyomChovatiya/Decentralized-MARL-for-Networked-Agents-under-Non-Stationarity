"""
Executes Priorities 1-4 End-to-End Research Suite:
- Priority 1: Verify unified training of IPPO, Networked-PPO, and BayesG
- Priority 2: Unshocked Normal-Operation Baseline (0-hop IPPO vs k-hop GNN vs BayesG)
- Priority 3: Non-Stationarity Shock Benchmark (Random Shock & Scale-Free Hub vs Periphery Shock)
- Priority 4: BayesG Dynamic Ego-Graph Edge Pruning Visualization ("The Money Plot")
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.experiments.run_experiment import run_single_experiment, get_device
from src.evaluation.plot_results import (
    plot_multi_algorithm_curves,
    plot_hub_vs_periphery_comparison,
    plot_bayesg_edge_pruning,
)


def main():
    device = "cpu"
    print("=" * 70)
    print(f"[SUITE] Running Full Priority 1-4 Experimental Suite (Device: {device}, XPU Verified: {get_device()})")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # PRIORITY 2: Normal Operation Baseline (No Shocks)
    # -------------------------------------------------------------------------
    print("\n>>> [Priority 2] Running Unshocked Baseline (IPPO vs Networked-PPO vs BayesG)...")
    p2_dir = "results/priority2_no_shock"
    os.makedirs(p2_dir, exist_ok=True)
    p2_results = {}
    for algo in ["ippo", "networked_ppo", "bayesg"]:
        metrics = run_single_experiment(
            topology="ring",
            num_agents=10,
            algorithm=algo,
            k_hops=2,
            hidden_dim=64,
            max_steps=30,
            total_episodes=90,
            lr=8e-4,
            shock_step=None,
            shock_fraction=0.0,
            selection_mode="none",
            seed=42,
            output_dir=os.path.join(p2_dir, algo),
            device=device,
            verbose=False,
        )
        p2_results[algo] = metrics
        print(
            f"   [P2 - {algo.upper():13s}] Mean Return: {metrics['mean_return']:6.2f} | "
            f"Max Return: {metrics['max_return']:6.2f} | "
            f"Conv. Ep: {metrics['convergence_episode']:3d} | "
            f"Comm Cost: {metrics['avg_communication_cost']:.2f}"
        )

    plot_multi_algorithm_curves(
        p2_results,
        title="Priority 2: Normal Operation Baseline (No Shock, Ring N=10)",
        shock_step=None,
        save_path=os.path.join(p2_dir, "no_shock_comparison.png"),
    )
    with open(os.path.join(p2_dir, "summary.json"), "w") as f:
        json.dump(p2_results, f, indent=2)

    # -------------------------------------------------------------------------
    # PRIORITY 3A: Non-Stationarity Shock Benchmark (Ring Graph, Random Shock)
    # -------------------------------------------------------------------------
    print("\n>>> [Priority 3A] Running Non-Stationarity Shock Benchmark (t_c = 50)...")
    p3a_dir = "results/priority3_random_shock"
    os.makedirs(p3a_dir, exist_ok=True)
    p3a_results = {}
    shock_ep = 50
    for algo in ["ippo", "networked_ppo", "bayesg"]:
        metrics = run_single_experiment(
            topology="ring",
            num_agents=10,
            algorithm=algo,
            k_hops=2,
            hidden_dim=64,
            max_steps=30,
            total_episodes=110,
            lr=8e-4,
            shock_step=shock_ep,
            shock_fraction=0.3,
            selection_mode="random",
            shock_behavior="adversarial",
            seed=42,
            output_dir=os.path.join(p3a_dir, algo),
            device=device,
            verbose=False,
        )
        p3a_results[algo] = metrics
        print(
            f"   [P3A - {algo.upper():13s}] Mean Return: {metrics['mean_return']:6.2f} | "
            f"Instability: {metrics['instability_index']:5.2f} | "
            f"Adapt Latency: {metrics['adaptation_latency']:2d} eps | "
            f"Comm Cost: {metrics['avg_communication_cost']:.2f}"
        )

    plot_multi_algorithm_curves(
        p3a_results,
        title="Priority 3A: Resilience to Mid-Training Policy Shock (Ring N=10, t_c=50)",
        shock_step=shock_ep,
        save_path=os.path.join(p3a_dir, "shock_comparison.png"),
    )
    with open(os.path.join(p3a_dir, "summary.json"), "w") as f:
        json.dump(p3a_results, f, indent=2)

    # -------------------------------------------------------------------------
    # PRIORITY 3B: Scale-Free Network — Hub Shock vs. Periphery Shock
    # -------------------------------------------------------------------------
    print("\n>>> [Priority 3B] Running Scale-Free Hub vs. Periphery Shock Study (N=12)...")
    p3b_dir = "results/priority3_hub_vs_periphery"
    os.makedirs(p3b_dir, exist_ok=True)
    p3b_results = {}
    scenarios = [
        ("NetPPO (Hub Shock)", "networked_ppo", "hubs"),
        ("NetPPO (Boundary Shock)", "networked_ppo", "boundary"),
        ("BayesG (Hub Shock)", "bayesg", "hubs"),
        ("BayesG (Boundary Shock)", "bayesg", "boundary"),
    ]
    for label, algo, mode in scenarios:
        safe_name = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        metrics = run_single_experiment(
            topology="scale_free",
            num_agents=12,
            algorithm=algo,
            k_hops=2,
            hidden_dim=64,
            max_steps=30,
            total_episodes=110,
            lr=8e-4,
            shock_step=shock_ep,
            shock_fraction=0.25,
            selection_mode=mode,
            shock_behavior="adversarial",
            seed=42,
            output_dir=os.path.join(p3b_dir, safe_name),
            device=device,
            verbose=False,
        )
        p3b_results[label] = metrics
        print(
            f"   [P3B - {label:23s}] Shocked Nodes: {metrics['shocked_nodes']} | "
            f"Mean Return: {metrics['mean_return']:6.2f} | "
            f"Instability: {metrics['instability_index']:5.2f} | "
            f"Shock Drop: {metrics['shock_drop']:5.2f} | "
            f"Adapt Latency: {metrics['adaptation_latency']:2d} eps"
        )

    plot_hub_vs_periphery_comparison(
        p3b_results,
        shock_step=shock_ep,
        save_path=os.path.join(p3b_dir, "hub_vs_periphery.png"),
    )
    with open(os.path.join(p3b_dir, "summary.json"), "w") as f:
        json.dump(p3b_results, f, indent=2)

    # -------------------------------------------------------------------------
    # PRIORITY 4: BayesG Edge Pruning Visualization ("The Money Plot")
    # -------------------------------------------------------------------------
    print("\n>>> [Priority 4] Generating BayesG Edge Pruning Heatmap & Graph Visualization...")
    p4_dir = "results/priority4_edge_pruning"
    os.makedirs(p4_dir, exist_ok=True)
    bayes_hub_metrics = p3b_results["BayesG (Hub Shock)"]

    adj_mat = np.array(bayes_hub_metrics["adj_matrix"], dtype=np.float64)
    pre_mask = np.array(bayes_hub_metrics["pre_shock_mask"], dtype=np.float64)
    post_mask = np.array(bayes_hub_metrics["post_shock_mask"], dtype=np.float64)
    shocked_nodes = bayes_hub_metrics["shocked_nodes"]

    # Compute mean edge weight to shocked vs unshocked neighbors before & after shock
    N = adj_mat.shape[0]
    eye = np.eye(N)
    phys_edges = (adj_mat > 0) & (eye == 0)
    shock_edge_mask = np.zeros_like(phys_edges, dtype=bool)
    for s in shocked_nodes:
        shock_edge_mask[:, s] = True
        shock_edge_mask[s, :] = True
    shock_phys = phys_edges & shock_edge_mask
    healthy_phys = phys_edges & (~shock_edge_mask)

    pre_shock_edge_mean = float(np.mean(pre_mask[shock_phys])) if np.any(shock_phys) else 0.0
    post_shock_edge_mean = float(np.mean(post_mask[shock_phys])) if np.any(shock_phys) else 0.0
    pre_healthy_edge_mean = float(np.mean(pre_mask[healthy_phys])) if np.any(healthy_phys) else 0.0
    post_healthy_edge_mean = float(np.mean(post_mask[healthy_phys])) if np.any(healthy_phys) else 0.0

    print(
        f"   [P4 - Edge Weights to Shocked Hubs {shocked_nodes}] "
        f"Pre-Shock: {pre_shock_edge_mean:.3f} -> Post-Shock: {post_shock_edge_mean:.3f}"
    )
    print(
        f"   [P4 - Edge Weights between Healthy Nodes] "
        f"Pre-Shock: {pre_healthy_edge_mean:.3f} -> Post-Shock: {post_healthy_edge_mean:.3f}"
    )

    plot_bayesg_edge_pruning(
        adj_matrix=adj_mat,
        pre_mask=pre_mask,
        post_mask=post_mask,
        shocked_nodes=shocked_nodes,
        save_path=os.path.join(p4_dir, "bayesg_edge_pruning.png"),
    )

    print("\n" + "=" * 70)
    print("[COMPLETE] All Priorities 1-4 Finished Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
