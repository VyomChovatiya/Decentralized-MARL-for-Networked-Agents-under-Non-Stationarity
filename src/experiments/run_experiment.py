"""
Universal Experiment Runner (Track 3)
Entry point for training, evaluating, and bench-testing networked MARL algorithms.
"""
from typing import Dict, Any
import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm

from src.environments.graphs import generate_topology, get_adjacency_matrix, compute_graph_metrics
from src.environments.toy_consensus import NetworkedConsensusEnv
from src.algorithms.ippo import IPPOTrainer
from src.experiments.shock_protocol import NonStationarityShockController
from src.evaluation.metrics import compute_experiment_metrics
from src.evaluation.plot_results import plot_learning_curve


def parse_args():
    parser = argparse.ArgumentParser(description="Networked MARL Experiment Runner")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--topology", type=str, default=None, help="Override network topology")
    parser.add_argument("--num_agents", type=int, default=None, help="Override number of agents")
    parser.add_argument("--algorithm", type=str, default=None, help="Algorithm: ippo, networked_ppo, bayesg")
    parser.add_argument("--total_episodes", type=int, default=None, help="Total training episodes")
    parser.add_argument("--shock_step", type=int, default=None, help="Episode when policy shock is triggered")
    parser.add_argument("--output_dir", type=str, default="results/run_latest", help="Output directory")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Command-line overrides
    topology = args.topology or cfg.get("env", {}).get("topology", "ring")
    num_agents = args.num_agents or cfg.get("env", {}).get("num_agents", 10)
    total_episodes = args.total_episodes or cfg.get("train", {}).get("total_episodes", 150)
    shock_step = args.shock_step if args.shock_step is not None else cfg.get("shock", {}).get("step", 75)
    seed = cfg.get("seed", 42)

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"🚀 Launching Decentralized MARL Experiment")
    print(f"   Topology: {topology} | Agents: {num_agents} | Device: {device}")
    print(f"   Episodes: {total_episodes} | Shock Episode: {shock_step}")
    print("=" * 60)

    # 1. Environment & Graph Setup
    graph = generate_topology(topology, num_agents, seed=seed)
    graph_stats = compute_graph_metrics(graph)
    print(f"📊 Graph Metrics: Diameter={graph_stats['diameter']}, Alg. Connectivity={graph_stats['algebraic_connectivity']:.4f}")

    env = NetworkedConsensusEnv(
        num_agents=num_agents,
        topology_type=topology,
        max_steps=cfg.get("env", {}).get("max_steps", 40),
        graph=graph,
        seed=seed,
    )

    # 2. Shock Controller
    shock_controller = NonStationarityShockController(
        shock_step=shock_step,
        fraction=cfg.get("shock", {}).get("fraction", 0.3),
        selection_mode=cfg.get("shock", {}).get("selection_mode", "random"),
        behavior=cfg.get("shock", {}).get("behavior", "adversarial"),
        graph=graph,
        num_agents=num_agents,
        seed=seed,
    )
    print(f"⚡ Shock targets: {len(shock_controller.shocked_agents)} nodes {shock_controller.shocked_nodes}")

    # 3. Trainer Setup (Default to IPPO baseline)
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    trainer = IPPOTrainer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        lr=float(cfg.get("train", {}).get("lr", 3e-4)),
        gamma=float(cfg.get("train", {}).get("gamma", 0.99)),
        device=device,
    )

    episode_returns = []

    # 4. Training Loop
    progress_bar = tqdm(range(1, total_episodes + 1), desc="Training")
    for ep in progress_bar:
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False

        while not done:
            # Multi-agent action selection
            actions = {}
            log_probs = {}
            vals = {}
            for agent_id in env.agents:
                obs_t = torch.tensor(obs[agent_id], dtype=torch.float32, device=device)
                act, log_prob, val = trainer.select_action(obs_t)
                actions[agent_id] = act.item()
                log_probs[agent_id] = log_prob
                vals[agent_id] = val

            # Apply mid-training non-stationarity shock
            effective_actions = shock_controller.apply_shock_actions(ep, actions)

            # Step environment
            next_obs, rewards, terms, truncs, info = env.step(effective_actions)
            step_r = sum(rewards.values()) / num_agents
            ep_reward += step_r

            done = any(terms.values()) or any(truncs.values())

            # Store transitions for learner
            for agent_id in env.agents:
                obs_t = torch.tensor(obs[agent_id], dtype=torch.float32)
                act_t = torch.tensor(actions[agent_id], dtype=torch.int64)
                trainer.store_transition(
                    obs_t,
                    act_t,
                    log_probs[agent_id].cpu(),
                    rewards[agent_id],
                    terms[agent_id] or truncs[agent_id],
                    vals[agent_id].cpu(),
                )

            obs = next_obs

        # Update policy
        trainer.update()
        episode_returns.append(ep_reward)

        if ep % 10 == 0 or ep == total_episodes:
            recent_mean = np.mean(episode_returns[-10:])
            progress_bar.set_postfix({"Recent Return": f"{recent_mean:.2f}"})

    # 5. Compute Final Metrics & Save Artifacts
    metrics = compute_experiment_metrics(episode_returns, shock_step=shock_step)
    metrics["graph_stats"] = graph_stats
    metrics["returns"] = episode_returns
    metrics["shock_step"] = shock_step

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    plot_path = os.path.join(args.output_dir, "learning_curve.png")
    plot_learning_curve(
        episode_returns,
        title=f"Decentralized MARL ({topology.capitalize()} Topology, N={num_agents})",
        shock_step=shock_step,
        save_path=plot_path,
    )

    print("\n" + "=" * 60)
    print("✅ Experiment Run Complete!")
    print(f"   Mean Return: {metrics['mean_return']:.3f}")
    print(f"   Instability Index: {metrics['instability_index']:.3f}")
    print(f"   Adaptation Latency: {metrics['adaptation_latency']} episodes")
    print(f"   Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
