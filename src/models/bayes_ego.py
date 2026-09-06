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
        temperature: float = 1.0,
        prior_prob: float = 0.3,
    ):
        super().__init__()
        self.temperature = temperature
        self.prior_prob = prior_prob

        # Pairwise edge relevance scoring network
        self.edge_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Infers latent edge mask Z and computes KL divergence to prior.

        Args:
            node_features: (num_agents, feat_dim) or (batch, num_agents, feat_dim)
            adj_matrix: (num_agents, num_agents) physical topology mask
            hard: if True, binarize output (for evaluation)

        Returns:
            edge_mask: (num_agents, num_agents) continuous or binary mask
            kl_loss: scalar tensor of KL divergence for ELBO loss
        """
        is_batched = node_features.dim() == 3
        if not is_batched:
            node_features = node_features.unsqueeze(0)

        batch_size, num_agents, feat_dim = node_features.shape

        # Construct pairwise representations (batch, N, N, 2 * feat_dim)
        feat_i = node_features.unsqueeze(2).expand(-1, -1, num_agents, -1)
        feat_j = node_features.unsqueeze(1).expand(-1, num_agents, -1, -1)
        pair_feat = torch.cat([feat_i, feat_j], dim=-1)

        # Compute edge logits (batch, N, N)
        logits = self.edge_mlp(pair_feat).squeeze(-1)
        probs = torch.sigmoid(logits)

        # Apply physical topology constraint: only existing edges can be active
        adj_expanded = adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        valid_edge_mask = (adj_expanded > 0).float()

        if self.training:
            # Gumbel-Softmax / Concrete relaxation for differentiable sampling
            gumbels = -torch.empty_like(logits).exponential_().log()
            gumbels_neg = -torch.empty_like(logits).exponential_().log()
            y_soft = torch.sigmoid((logits + gumbels - gumbels_neg) / self.temperature)

            if hard:
                y_hard = (y_soft > 0.5).float()
                sampled_mask = (y_hard - y_soft).detach() + y_soft
            else:
                sampled_mask = y_soft
        else:
            sampled_mask = (probs > 0.5).float() if hard else probs

        # Constrain to valid physical edges
        active_mask = sampled_mask * valid_edge_mask

        # Compute KL divergence: KL(Bernoulli(probs) || Bernoulli(p0))
        p0 = torch.tensor(self.prior_prob, device=probs.device)
        eps = 1e-7
        p_clamped = probs.clamp(eps, 1.0 - eps)
        kl = (
            p_clamped * torch.log(p_clamped / p0)
            + (1.0 - p_clamped) * torch.log((1.0 - p_clamped) / (1.0 - p0))
        )
        # Average KL only over existing physical graph edges
        kl_loss = (kl * valid_edge_mask).sum() / valid_edge_mask.sum().clamp(min=1.0)

        if not is_batched:
            active_mask = active_mask.squeeze(0)

        return active_mask, kl_loss
