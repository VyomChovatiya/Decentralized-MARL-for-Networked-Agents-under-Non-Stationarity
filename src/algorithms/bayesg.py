"""
BayesG Algorithm (Track 2)
Decentralized Actor-Critic with Bayesian Ego-Graph Variational Inference.
Based on Wei Duan et al., NeurIPS 2025.
"""
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.actors import DecentralizedMLPActor
from src.models.critics import DecentralizedMLPCritic
from src.models.gnn_comm import GNNCommunicationLayer
from src.models.bayes_ego import BayesGEgoInference


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
        elbo_beta: float = 0.05,
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
        self.device = torch.device(device)

        # 1. Latent Ego-Graph Variational Inference
        self.bayes_ego = BayesGEgoInference(feat_dim=obs_dim).to(self.device)
        # 2. GNN message passing over gated edges
        self.gnn = GNNCommunicationLayer(obs_dim, gnn_dim, k_hops=k_hops).to(self.device)
        # 3. Policy and Value functions
        self.actor = DecentralizedMLPActor(gnn_dim, act_dim).to(self.device)
        self.critic = DecentralizedMLPCritic(gnn_dim).to(self.device)

        all_params = (
            list(self.bayes_ego.parameters())
            + list(self.gnn.parameters())
            + list(self.actor.parameters())
        )
        self.actor_optim = optim.Adam(all_params, lr=lr)
        self.critic_optim = optim.Adam(
            list(self.gnn.parameters()) + list(self.critic.parameters()), lr=lr
        )

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
        edge_mask, kl_loss = self.bayes_ego(obs_tensor, adj_tensor, hard=eval_mode)
        node_emb = self.gnn(obs_tensor, adj_tensor, edge_mask=edge_mask)

        with torch.no_grad():
            actions, log_probs = self.actor.get_action(node_emb)
            values = self.critic(node_emb)

        return actions, log_probs, values, edge_mask, kl_loss
