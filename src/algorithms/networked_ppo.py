"""
Networked PPO with GNN Communication (Track 2)
Agents exchange observation embeddings over k-hop graph neighborhoods.
"""
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.actors import DecentralizedMLPActor
from src.models.critics import DecentralizedMLPCritic
from src.models.gnn_comm import GNNCommunicationLayer
from .ippo import RolloutBuffer


class NetworkedPPOTrainer:
    """
    Networked Actor-Critic with static k-hop GNN communication.
    Agents aggregate messages from their fixed k-hop graph neighborhood.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        k_hops: int = 1,
        gnn_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        epochs: int = 4,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.k_hops = k_hops
        self.device = torch.device(device)

        # GNN encoder + Actor-Critic heads
        self.gnn = GNNCommunicationLayer(obs_dim, gnn_dim, k_hops=k_hops).to(self.device)
        self.actor = DecentralizedMLPActor(gnn_dim, act_dim, hidden_dim=gnn_dim).to(self.device)
        self.critic = DecentralizedMLPCritic(gnn_dim, hidden_dim=gnn_dim).to(self.device)

        all_params = (
            list(self.gnn.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters())
        )
        self.optimizer = optim.Adam(all_params, lr=lr)

        self.buffer = RolloutBuffer()
        self.last_comm_cost = 1.0

    def encode_and_act(self, obs_tensor: torch.Tensor, adj_tensor: torch.Tensor):
        """
        Args:
            obs_tensor: (num_agents, obs_dim)
            adj_tensor: (num_agents, num_agents)
        """
        with torch.no_grad():
            node_emb = self.gnn(obs_tensor, adj_tensor)
            actions, log_probs = self.actor.get_action(node_emb)
            values = self.critic(node_emb)
        return actions, log_probs, values, node_emb

    def select_action(
        self,
        obs_tensor: torch.Tensor,
        adj_tensor: torch.Tensor,
        eval_mode: bool = False,
    ):
        actions, log_probs, values, _ = self.encode_and_act(obs_tensor, adj_tensor)
        self.last_comm_cost = 1.0
        return actions, log_probs, values

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

        # Shapes: (T, N, obs_dim), (T, N), (T, N), (T, N)
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
        total_entropy = 0.0
        all_params = (
            list(self.gnn.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters())
        )

        for _ in range(self.epochs):
            # Re-run GNN over batch of timesteps (T, N, obs_dim) -> (T, N, gnn_dim)
            node_emb = self.gnn(obs_t, adj_t)
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
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 0.5)
            self.optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()

        self.buffer.clear()

        return {
            "actor_loss": total_actor_loss / self.epochs,
            "critic_loss": total_critic_loss / self.epochs,
            "entropy": total_entropy / self.epochs,
            "comm_cost": 1.0,
        }
