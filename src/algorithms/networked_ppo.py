"""
Networked PPO with GNN Communication (Track 2)
Agents exchange observation embeddings over k-hop graph neighborhoods.
"""
from typing import Dict, List, Any
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
    Networked Actor-Critic with k-hop GNN communication.
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
        self.device = torch.device(device)

        # GNN encoder
        self.gnn = GNNCommunicationLayer(obs_dim, gnn_dim, k_hops=k_hops).to(self.device)
        self.actor = DecentralizedMLPActor(gnn_dim, act_dim).to(self.device)
        self.critic = DecentralizedMLPCritic(gnn_dim).to(self.device)

        params = list(self.gnn.parameters()) + list(self.actor.parameters())
        self.actor_optim = optim.Adam(params, lr=lr)
        self.critic_optim = optim.Adam(list(self.gnn.parameters()) + list(self.critic.parameters()), lr=lr)

        self.buffer = RolloutBuffer()

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
