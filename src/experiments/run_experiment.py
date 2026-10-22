"""
Universal Experiment Runner (Track 3)
Entry point for training, evaluating, and bench-testing networked MARL algorithms.
Supports IPPO, Networked-PPO (k-hop GNN), and BayesG (Bayesian Ego-Graph Inference).
"""
from typing import Dict, Any, Optional
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
from src.algorithms.networked_ppo import NetworkedPPOTrainer
from src.algorithms.bayesg import BayesGTrainer
from src.experiments.shock_protocol import NonStationarityShockController
from src.evaluation.metrics import compute_experiment_metrics
from src.evaluation.plot_results import plot_learning_curve


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Networked MARL Experiment Runner")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--topology", type=str, default=None, help="Override network topology")
    parser.add_argument("--num_agents", type=int, default=None, help="Override number of agents")
    parser.add_argument("--algorithm", type=str, default=None, help="Algorithm: ippo, networked_ppo, bayesg")
    parser.add_argument("--k_hops", type=int, default=None, help="Communication radius k")
    parser.add_argument("--total_episodes", type=int, default=None, help="Total training episodes")
    parser.add_argument("--shock_step", type=int, default=None, help="Episode when policy shock is triggered (-1 for no shock)")
    parser.add_argument("--shock_fraction", type=float, default=None, help="Fraction of nodes shocked")
    parser.add_argument("--selection_mode", type=str, default=None, help="Shock node selection: random, hubs, boundary, none")
    parser.add_argument("--output_dir", type=str, default="results/run_latest", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def build_trainer(
    algorithm: str,
    obs_dim: int,
    act_dim: int,
    hidden_dim: int,
    k_hops: int,
    lr: float,
    gamma: float,
    gae_lambda: float,
    clip_ratio: float,
    epochs: int,
    device: str,
):
    algo = algorithm.lower()
    if algo == "ippo":
        return IPPOTrainer(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            epochs=epochs,
            device=device,
        )
    elif algo in ["networked_ppo", "gnn", "net_ppo"]:
        return NetworkedPPOTrainer(
            obs_dim=obs_dim,
            act_dim=act_dim,
            k_hops=k_hops,
            gnn_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            epochs=epochs,
            device=device,
        )
    elif algo == "bayesg":
        return BayesGTrainer(
            obs_dim=obs_dim,
            act_dim=act_dim,
            gnn_dim=hidden_dim,
            k_hops=k_hops,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            epochs=epochs,
            device=device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from 'ippo', 'networked_ppo', 'bayesg'.")


def run_single_experiment(
    topology: str = "ring",
    num_agents: int = 10,
    algorithm: str = "ippo",
    k_hops: int = 1,
    hidden_dim: int = 64,
    max_steps: int = 35,
    total_episodes: int = 120,
    lr: float = 1e-3,
    gamma: float = 0.98,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    epochs: int = 4,
    shock_step: Optional[int] = 60,
    shock_fraction: float = 0.3,
    selection_mode: str = "random",
    shock_behavior: str = "adversarial",
    seed: int = 42,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    if device is None:
        device = get_device()

    torch.manual_seed(seed)
    np.random.seed(seed)

    if shock_step is not None and shock_step < 0:
        shock_step = None

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("[LAUNCH] Launching Decentralized MARL Experiment")
        print(f"   Algorithm: {algorithm.upper()} | Topology: {topology} | Agents: {num_agents} | Device: {device}")
        print(f"   Episodes: {total_episodes} | Shock Episode: {shock_step} ({selection_mode})")
        print("=" * 60)

    # 1. Environment & Graph Setup
    graph = generate_topology(topology, num_agents, seed=seed)
    graph_stats = compute_graph_metrics(graph)
    if verbose:
        print(
            f"[GRAPH] Metrics: Diameter={graph_stats['diameter']}, "
            f"Alg. Connectivity={graph_stats['algebraic_connectivity']:.4f}"
        )

    # Physical adjacency matrix with self-loops
    adj_raw = get_adjacency_matrix(graph, normalized=False)
    adj_with_self = adj_raw + np.eye(num_agents, dtype=np.float32)
    adj_tensor = torch.tensor(adj_with_self, dtype=torch.float32, device=device)

    env = NetworkedConsensusEnv(
        num_agents=num_agents,
        topology_type=topology,
        max_steps=max_steps,
        graph=graph,
        seed=seed,
    )

    # 2. Shock Controller
    shock_controller = NonStationarityShockController(
        shock_step=shock_step,
        fraction=shock_fraction,
        selection_mode=selection_mode,
        behavior=shock_behavior,
        graph=graph,
        num_agents=num_agents,
        seed=seed,
    )
    if verbose:
        print(f"[SHOCK] Targets: {len(shock_controller.shocked_agents)} nodes {shock_controller.shocked_nodes}")

    # 3. Trainer Setup
    trainer = build_trainer(
        algorithm=algorithm,
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        hidden_dim=hidden_dim,
        k_hops=k_hops,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_ratio=clip_ratio,
        epochs=epochs,
        device=device,
    )

    episode_returns = []
    comm_costs = []
    pre_shock_masks = []
    post_shock_masks = []

    active_agent_ids = [
        a for a in env.agents if a not in shock_controller.shocked_agents
    ]
    if len(active_agent_ids) == 0:
        active_agent_ids = list(env.agents)

    # 4. Training Loop
    iterator = tqdm(range(1, total_episodes + 1), desc=f"Train {algorithm.upper()}", disable=not verbose)
    for ep in iterator:
        obs, _ = env.reset(seed=seed * 1000 + ep)
        obs = shock_controller.apply_shock_observations(ep, obs)

        ep_reward = 0.0
        ep_comm = []
        done = False

        while not done:
            # Stack observations into (num_agents, obs_dim)
            obs_np = np.stack([obs[a] for a in env.agents], axis=0)
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)

            out = trainer.select_action(obs_t, adj_tensor)
            actions_t, log_probs_t, vals_t = out[0], out[1], out[2]

            actions_dict = {
                agent_id: int(actions_t[i].item())
                for i, agent_id in enumerate(env.agents)
            }

            effective_actions = shock_controller.apply_shock_actions(ep, actions_dict)
            active_shock_set = shock_controller.shocked_agents if shock_controller.is_active else None

            next_obs, rewards, terms, truncs, _ = env.step(
                effective_actions, shocked_agents=active_shock_set
            )
            next_obs = shock_controller.apply_shock_observations(ep, next_obs)

            # Evaluate cooperative episode return across operational agents
            if shock_controller.is_active:
                step_r = sum(rewards[a] for a in active_agent_ids) / len(active_agent_ids)
            else:
                step_r = sum(rewards.values()) / num_agents
            ep_reward += step_r

            done = any(terms.values()) or any(truncs.values())

            rewards_vec = np.array([rewards[a] for a in env.agents], dtype=np.float32)
            trainer.store_transition(
                obs_t,
                actions_t,
                log_probs_t,
                rewards_vec,
                done,
                vals_t,
                adj=adj_tensor,
            )

            ep_comm.append(getattr(trainer, "last_comm_cost", 0.0))
            if hasattr(trainer, "last_edge_mask") and trainer.last_edge_mask is not None:
                if shock_step is not None and ep < shock_step and ep >= max(1, shock_step - 15):
                    pre_shock_masks.append(trainer.last_edge_mask.copy())
                elif shock_step is not None and ep >= shock_step + 5:
                    post_shock_masks.append(trainer.last_edge_mask.copy())

            obs = next_obs

        trainer.update(adj_tensor) if hasattr(trainer, "k_hops") else trainer.update()
        episode_returns.append(float(ep_reward))
        comm_costs.append(float(np.mean(ep_comm)) if len(ep_comm) > 0 else 0.0)

        if verbose and (ep % 10 == 0 or ep == total_episodes):
            recent_mean = np.mean(episode_returns[-10:])
            iterator.set_postfix({"Return": f"{recent_mean:.2f}", "Comm": f"{comm_costs[-1]:.2f}"})

    # 5. Compute Final Metrics & Save Artifacts
    metrics = compute_experiment_metrics(
        episode_returns, shock_step=shock_step, comm_costs=comm_costs
    )
    metrics["algorithm"] = algorithm
    metrics["topology"] = topology
    metrics["num_agents"] = num_agents
    metrics["graph_stats"] = graph_stats
    metrics["returns"] = episode_returns
    metrics["comm_costs"] = comm_costs
    metrics["shock_step"] = shock_step
    metrics["shocked_nodes"] = shock_controller.shocked_nodes
    metrics["selection_mode"] = selection_mode

    if len(pre_shock_masks) > 0:
        metrics["pre_shock_mask"] = np.mean(pre_shock_masks, axis=0).tolist()
    if len(post_shock_masks) > 0:
        metrics["post_shock_mask"] = np.mean(post_shock_masks, axis=0).tolist()
    metrics["adj_matrix"] = adj_raw.tolist()

    if output_dir:
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        plot_path = os.path.join(output_dir, "learning_curve.png")
        plot_learning_curve(
            episode_returns,
            title=f"{algorithm.upper()} on {topology.capitalize()} Topology (N={num_agents})",
            shock_step=shock_step,
            save_path=plot_path,
        )

    if verbose:
        print("\n" + "=" * 60)
        print("[DONE] Experiment Run Complete!")
        print(f"   Algorithm: {algorithm.upper()} | Topology: {topology}")
        print(f"   Mean Return: {metrics['mean_return']:.3f} | Max Return: {metrics['max_return']:.3f}")
        print(f"   Instability Index: {metrics['instability_index']:.3f}")
        print(f"   Adaptation Latency: {metrics['adaptation_latency']} episodes")
        print(f"   Avg Comm Cost: {metrics['avg_communication_cost']:.3f}")
        if output_dir:
            print(f"   Results saved to: {output_dir}")
        print("=" * 60)

    return metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)

    topology = args.topology or cfg.get("env", {}).get("topology", "ring")
    num_agents = args.num_agents or cfg.get("env", {}).get("num_agents", 10)
    max_steps = cfg.get("env", {}).get("max_steps", 35)

    algorithm = args.algorithm or cfg.get("algorithm", {}).get("name", "ippo")
    hidden_dim = cfg.get("algorithm", {}).get("hidden_dim", 64)
    k_hops = args.k_hops or cfg.get("algorithm", {}).get("k_hops", 1)

    total_episodes = args.total_episodes or cfg.get("train", {}).get("total_episodes", 120)
    lr = float(cfg.get("train", {}).get("lr", 1e-3))
    gamma = float(cfg.get("train", {}).get("gamma", 0.98))
    gae_lambda = float(cfg.get("train", {}).get("gae_lambda", 0.95))
    clip_ratio = float(cfg.get("train", {}).get("clip_ratio", 0.2))
    epochs = int(cfg.get("train", {}).get("epochs", 4))

    shock_step = args.shock_step if args.shock_step is not None else cfg.get("shock", {}).get("step", 60)
    shock_fraction = args.shock_fraction if args.shock_fraction is not None else cfg.get("shock", {}).get("fraction", 0.3)
    selection_mode = args.selection_mode or cfg.get("shock", {}).get("selection_mode", "random")
    shock_behavior = cfg.get("shock", {}).get("behavior", "adversarial")
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)

    run_single_experiment(
        topology=topology,
        num_agents=num_agents,
        algorithm=algorithm,
        k_hops=k_hops,
        hidden_dim=hidden_dim,
        max_steps=max_steps,
        total_episodes=total_episodes,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_ratio=clip_ratio,
        epochs=epochs,
        shock_step=shock_step,
        shock_fraction=shock_fraction,
        selection_mode=selection_mode,
        shock_behavior=shock_behavior,
        seed=seed,
        output_dir=args.output_dir,
        verbose=True,
    )


if __name__ == "__main__":
    main()
