"""
Decentralized Critic Networks (Track 2)
"""
import torch
import torch.nn as nn


class DecentralizedMLPCritic(nn.Module):
    """
    Decentralized Critic Network.
    Maps local observation -> predicted state value V(o_i).
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns value prediction: shape (batch_size, 1) or (batch_size,)"""
        return self.net(obs).squeeze(-1)
