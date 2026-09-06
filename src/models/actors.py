"""
Decentralized Policy Networks (Track 2)
"""
from typing import Tuple
import torch
import torch.nn as nn
from torch.distributions import Categorical


class DecentralizedMLPActor(nn.Module):
    """
    Decentralized Actor Network.
    Maps local observation -> categorical action distribution.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        return Categorical(logits=logits)

    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob)"""
        dist = self(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log_prob and entropy for given actions"""
        dist = self(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy
