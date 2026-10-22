"""
BayesG Algorithm (Track 2)
Decentralized Actor-Critic with Bayesian Ego-Graph Variational Inference.
Based on Wei Duan et al., NeurIPS 2025.
"""
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.actors import DecentralizedMLPActor
from src.models.critics import DecentralizedMLPCritic
from src.models.gnn_comm import GNNCommunicationLayer
from src.models.bayes_ego import BayesGEgoInference
from .ippo import RolloutBuffer


class BayesGTrainer:
    """
    BayesG: Joint policy learning and sparse ego-graph variational inference.
    Dynamically learns which neighbors to listen to, filtering non-stationary neighbors.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        gnn_dim: int = 64,
        k_hops: int = 1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        elbo_beta: float = 0.08,
        epochs: int = 4,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.elbo_beta = elbo_beta
        self.epochs = epochs
        self.k_hops = k_hops
        self.device = torch.device(device)

        # 1. Latent Ego-Graph Variational Inference
        self.bayes_ego = BayesGEgoInference(feat_dim=obs_dim, hidden_dim=gnn_dim).to(self.device)
        # 2. GNN message passing over gated edges
        self.gnn = GNNCommunicationLayer(obs_dim, gnn_dim, k_hops=k_hops).to(self.device)
        # 3. Policy and Value functions
        self.actor = DecentralizedMLPActor(gnn_dim, act_dim, hidden_dim=gnn_dim).to(self.device)
        self.critic = DecentralizedMLPCritic(gnn_dim, hidden_dim=gnn_dim).to(self.device)

        self.all_params = (
            list(self.bayes_ego.parameters())
            + list(self.gnn.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters())
        )
        self.optimizer = optim.Adam(self.all_params, lr=lr)

        self.buffer = RolloutBuffer()
        self.last_edge_mask: Optional[np.ndarray] = None
        self.last_comm_cost: float = 1.0

    def _compute_comm_cost(self, edge_mask: torch.Tensor, adj_tensor: torch.Tensor) -> float:
        N = adj_tensor.shape[0]
        eye = torch.eye(N, device=adj_tensor.device)
        off_diag_phys = (adj_tensor > 0).float() * (1.0 - eye)
        total_phys_edges = off_diag_phys.sum().clamp(min=1.0)
        if edge_mask.dim() == 3:
            active_edges = (edge_mask * off_diag_phys.unsqueeze(0)).sum(dim=(-2, -1)).mean()
        else:
            active_edges = (edge_mask * off_diag_phys).sum()
        return float((active_edges / total_phys_edges).item())

    def select_action(
        self,
        obs_tensor: torch.Tensor,
        adj_tensor: torch.Tensor,
        eval_mode: bool = False,
    ):
        """
        Infers ego-graph communication mask, performs message passing, and samples actions.
        """
        self.bayes_ego.train(not eval_mode)
        with torch.no_grad():
            edge_mask, kl_loss = self.bayes_ego(obs_tensor, adj_tensor, hard=eval_mode)
            node_emb = self.gnn(obs_tensor, adj_tensor, edge_mask=edge_mask)
            actions, log_probs = self.actor.get_action(node_emb)
            values = self.critic(node_emb)

            # Also compute deterministic probability mask for clean logging/visualization
            self.bayes_ego.eval()
            det_mask, _ = self.bayes_ego(obs_tensor, adj_tensor, hard=False)
            self.bayes_ego.train(not eval_mode)

            self.last_edge_mask = det_mask.detach().cpu().numpy()
            self.last_comm_cost = self._compute_comm_cost(det_mask, adj_tensor)

        return actions, log_probs, values, edge_mask, kl_loss

    def store_transition(self, obs, action, log_prob, reward, done, val, adj=None):
        self.buffer.obs.append(obs.detach().cpu() if isinstance(obs, torch.Tensor) else torch.tensor(obs, dtype=torch.float32))
        self.buffer.actions.append(action.detach().cpu() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.int64))
        self.buffer.log_probs.append(log_prob.detach().cpu() if isinstance(log_prob, torch.Tensor) else torch.tensor(log_prob, dtype=torch.float32))
        if isinstance(reward, torch.Tensor):
            self.buffer.rewards.append(reward.detach().cpu().numpy())
        else:
            self.buffer.rewards.append(np.asarray(reward, dtype=np.float32))
        self.buffer.dones.append(bool(done))
        self.buffer.values.append(val.detach().cpu() if isinstance(val, torch.Tensor) else torch.tensor(val, dtype=torch.float32))
        if adj is not None:
            self.buffer.adjs.append(adj.detach().cpu() if isinstance(adj, torch.Tensor) else torch.tensor(adj, dtype=torch.float32))

    def update(self, adj_tensor: Optional[torch.Tensor] = None) -> Dict[str, float]:
        if len(self.buffer.rewards) == 0:
            return {}

        obs_t = torch.stack(self.buffer.obs).to(self.device)
        actions_t = torch.stack(self.buffer.actions).to(self.device)
        old_log_probs_t = torch.stack(self.buffer.log_probs).to(self.device)
        values_t = torch.stack(self.buffer.values).to(self.device)
        rewards_np = np.stack(self.buffer.rewards, axis=0).astype(np.float32)
        dones = self.buffer.dones

        if adj_tensor is None and len(self.buffer.adjs) > 0:
            adj_t = self.buffer.adjs[0].to(self.device)
        elif adj_tensor is not None:
            adj_t = adj_tensor.to(self.device)
        else:
            num_agents = obs_t.shape[1]
            adj_t = torch.eye(num_agents, device=self.device)

        T = len(dones)
        values_np = values_t.detach().cpu().numpy()
        advantages = np.zeros_like(values_np, dtype=np.float32)
        last_gae = np.zeros_like(values_np[0], dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1 or dones[t]:
                next_val = np.zeros_like(values_np[0], dtype=np.float32)
            else:
                next_val = values_np[t + 1]

            non_terminal = 1.0 - float(dones[t])
            delta = rewards_np[t] + self.gamma * next_val * non_terminal - values_np[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values_np
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device).reshape(-1)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device).reshape(-1)

        if adv_t.numel() > 1:
            adv_t = (adv_t - adv_t.mean()) / torch.clamp(adv_t.std(), min=0.5)

        actions_flat = actions_t.reshape(-1)
        old_log_probs_flat = old_log_probs_t.reshape(-1)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_kl_loss = 0.0
        total_entropy = 0.0

        self.bayes_ego.train(True)
        for _ in range(self.epochs):
            # Re-infer ego-graph masks and GNN embeddings across batch (T, N, obs_dim)
            edge_mask, kl_loss = self.bayes_ego(obs_t, adj_t, hard=False)
            node_emb = self.gnn(obs_t, adj_t, edge_mask=edge_mask)
            emb_flat = node_emb.reshape(-1, node_emb.shape[-1])

            new_log_probs, entropy = self.actor.evaluate_actions(emb_flat, actions_flat)
            ratio = torch.exp(new_log_probs - old_log_probs_flat)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean()

            values_pred = self.critic(emb_flat)
            critic_loss = 0.5 * ((values_pred - returns_t) ** 2).mean()

            loss = (
                actor_loss
                + self.value_coef * critic_loss
                - self.entropy_coef * entropy.mean()
                + self.elbo_beta * kl_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.all_params, 0.5)
            self.optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_kl_loss += kl_loss.item()
            total_entropy += entropy.mean().item()

        self.buffer.clear()

        return {
            "actor_loss": total_actor_loss / self.epochs,
            "critic_loss": total_critic_loss / self.epochs,
            "kl_loss": total_kl_loss / self.epochs,
            "entropy": total_entropy / self.epochs,
            "comm_cost": self.last_comm_cost,
        }
