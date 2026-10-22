"""
Independent PPO (IPPO) Baseline (Track 2)
Decentralized Actor-Critic without communication.
"""
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.actors import DecentralizedMLPActor
from src.models.critics import DecentralizedMLPCritic


class RolloutBuffer:
    def __init__(self):
        self.obs: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[Any] = []
        self.dones: List[bool] = []
        self.values: List[torch.Tensor] = []
        self.adjs: List[torch.Tensor] = []

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.adjs.clear()


class IPPOTrainer:
    """
    Independent PPO for multi-agent systems.
    Each agent acts as an independent learner with parameter-shared policy & critic
    and zero inter-agent communication (0-hop).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 64,
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
        self.device = torch.device(device)

        self.actor = DecentralizedMLPActor(obs_dim, act_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic = DecentralizedMLPCritic(obs_dim, hidden_dim=hidden_dim).to(self.device)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = RolloutBuffer()
        self.last_comm_cost = 0.0

    def select_action(
        self,
        obs_tensor: torch.Tensor,
        adj_tensor: Optional[torch.Tensor] = None,
        eval_mode: bool = False,
    ):
        with torch.no_grad():
            action, log_prob = self.actor.get_action(obs_tensor)
            val = self.critic(obs_tensor)
        return action, log_prob, val

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

    def update(self) -> Dict[str, float]:
        if len(self.buffer.rewards) == 0:
            return {}

        # Shapes: either (T, N, ...) for multi-agent steps or (T, ...) for flat transitions
        obs_t = torch.stack(self.buffer.obs).to(self.device)
        actions_t = torch.stack(self.buffer.actions).to(self.device)
        old_log_probs_t = torch.stack(self.buffer.log_probs).to(self.device)
        values_t = torch.stack(self.buffer.values).to(self.device)
        rewards_np = np.stack(self.buffer.rewards, axis=0).astype(np.float32)
        dones = self.buffer.dones

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

        obs_flat = obs_t.reshape(-1, obs_t.shape[-1])
        actions_flat = actions_t.reshape(-1)
        old_log_probs_flat = old_log_probs_t.reshape(-1)

        if adv_t.numel() > 1:
            adv_t = (adv_t - adv_t.mean()) / torch.clamp(adv_t.std(), min=0.5)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.epochs):
            new_log_probs, entropy = self.actor.evaluate_actions(obs_flat, actions_flat)
            ratio = torch.exp(new_log_probs - old_log_probs_flat)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optim.step()

            values_pred = self.critic(obs_flat)
            critic_loss = 0.5 * ((values_pred - returns_t) ** 2).mean()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optim.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()

        self.buffer.clear()

        return {
            "actor_loss": total_actor_loss / self.epochs,
            "critic_loss": total_critic_loss / self.epochs,
            "entropy": total_entropy / self.epochs,
            "comm_cost": 0.0,
        }
