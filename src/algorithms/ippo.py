"""
Independent PPO (IPPO) Baseline (Track 2)
Decentralized Actor-Critic without communication.
"""
from typing import Dict, List, Any
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
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[torch.Tensor] = []

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()


class IPPOTrainer:
    """
    Independent PPO for multi-agent systems.
    Each agent acts as an independent learner with parameter-shared policy & critic.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
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

        self.actor = DecentralizedMLPActor(obs_dim, act_dim).to(self.device)
        self.critic = DecentralizedMLPCritic(obs_dim).to(self.device)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = RolloutBuffer()

    def select_action(self, obs_tensor: torch.Tensor):
        with torch.no_grad():
            action, log_prob = self.actor.get_action(obs_tensor)
            val = self.critic(obs_tensor)
        return action, log_prob, val

    def store_transition(self, obs, action, log_prob, reward, done, val):
        self.buffer.obs.append(obs)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(log_prob)
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)
        self.buffer.values.append(val)

    def update(self) -> Dict[str, float]:
        if len(self.buffer.rewards) == 0:
            return {}

        # Convert buffer to tensors
        obs_t = torch.stack(self.buffer.obs).to(self.device)
        actions_t = torch.stack(self.buffer.actions).to(self.device)
        old_log_probs_t = torch.stack(self.buffer.log_probs).to(self.device)
        values_t = torch.stack(self.buffer.values).to(self.device)
        rewards = self.buffer.rewards
        dones = self.buffer.dones

        # Compute GAE advantages and target returns
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = 0.0
            else:
                next_val = values_t[t + 1].item()

            non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_val * non_terminal - values_t[t].item()
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values_t.detach().cpu().numpy()
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # PPO training epochs
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.epochs):
            # Actor update
            new_log_probs, entropy = self.actor.evaluate_actions(obs_t, actions_t)
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optim.step()

            # Critic update
            values_pred = self.critic(obs_t)
            critic_loss = F_critic = 0.5 * ((values_pred - returns_t) ** 2).mean()

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
        }
