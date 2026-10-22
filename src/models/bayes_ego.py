"""
BayesG: Bayesian Ego-Graph Inference Module (Track 2)
Infers latent communication masks Z_ij over local ego-graphs using variational inference.
Based on Wei Duan et al., NeurIPS 2025.
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesGEgoInference(nn.Module):
    """
    Bayesian Ego-Graph Latent Mask Inference.
    Learns to prune or prioritize communication channels dynamically
    using Gumbel-Softmax variational inference and an ELBO objective.
    """

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 64,
        temperature: float = 0.8,
        prior_prob: float = 0.75,
    ):
        super().__init__()
        self.temperature = temperature
        self.prior_prob = prior_prob

        # Pairwise edge relevance scoring network: [h_i, h_j, |h_i - h_j|] -> logit
        self.edge_mlp = nn.Sequential(
            nn.Linear(feat_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        for layer in self.edge_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.3)
                nn.init.zeros_(layer.bias)
        # Initialize final bias positive so healthy edges start open (~0.90)
        nn.init.constant_(self.edge_mlp[-1].bias, 2.2)

    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Infers latent edge mask Z and computes ELBO KL + anomaly regularization loss.

        Args:
            node_features: (num_agents, feat_dim) or (batch, num_agents, feat_dim)
            adj_matrix: (num_agents, num_agents) physical topology mask
            hard: if True, binarize output (for evaluation)

        Returns:
            edge_mask: (num_agents, num_agents) or (batch, num_agents, num_agents) mask
            kl_loss: scalar tensor of KL divergence + relational anomaly penalty for ELBO
        """
        is_batched = node_features.dim() == 3
        if not is_batched:
            node_features = node_features.unsqueeze(0)

        batch_size, num_agents, _ = node_features.shape

        # Construct pairwise representations (batch, N, N, 3 * feat_dim)
        feat_i = node_features.unsqueeze(2).expand(-1, -1, num_agents, -1)
        feat_j = node_features.unsqueeze(1).expand(-1, num_agents, -1, -1)
        feat_diff = torch.abs(feat_i - feat_j)
        pair_feat = torch.cat([feat_i, feat_j, feat_diff], dim=-1)

        # Compute pairwise feature divergence so anomalous/shocked neighbors are pruned
        feature_dist = torch.mean(feat_diff, dim=-1)  # (batch, N, N)
        learned_logits = self.edge_mlp(pair_feat).squeeze(-1)
        logits = learned_logits - 4.5 * F.relu(feature_dist - 1.75)
        probs = torch.sigmoid(logits)

        # Apply physical topology constraint: only existing off-diagonal edges are inferred
        if adj_matrix.dim() == 2:
            adj_expanded = adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            adj_expanded = adj_matrix

        eye = torch.eye(num_agents, device=node_features.device).unsqueeze(0)
        off_diag_mask = (adj_expanded > 0).float() * (1.0 - eye)

        if self.training:
            # Low-variance Gumbel-Softmax / Binary Concrete relaxation
            u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
            logistic_noise = 0.2 * (torch.log(u) - torch.log(1.0 - u))
            y_soft = torch.sigmoid((logits + logistic_noise) / self.temperature)

            if hard:
                y_hard = (y_soft > 0.5).float()
                sampled_mask = (y_hard - y_soft).detach() + y_soft
            else:
                sampled_mask = y_soft
        else:
            sampled_mask = (probs > 0.5).float() if hard else probs

        # Constrain off-diagonal to valid physical edges and keep self-loops = 1
        active_mask = sampled_mask * off_diag_mask + eye * (adj_expanded > 0).float()

        # Compute KL divergence: KL(Bernoulli(probs) || Bernoulli(p0)) over off-diagonal edges
        p0 = torch.tensor(self.prior_prob, device=probs.device)
        eps = 1e-6
        p_clamped = probs.clamp(eps, 1.0 - eps)
        kl = (
            p_clamped * torch.log(p_clamped / p0)
            + (1.0 - p_clamped) * torch.log((1.0 - p_clamped) / (1.0 - p0))
        )

        # Compatibility loss: penalize high edge probability when neighbor feature divergence is extreme
        anomaly_penalty = probs * F.relu(feature_dist - 1.75)
        total_edge_reg = (kl + 3.0 * anomaly_penalty) * off_diag_mask
        kl_loss = total_edge_reg.sum() / off_diag_mask.sum().clamp(min=1.0)

        if not is_batched:
            active_mask = active_mask.squeeze(0)

        return active_mask, kl_loss
