"""
Graph Neural Network Communication Layer (Track 2)
Implements k-hop message passing over the agent graph topology.
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNCommunicationLayer(nn.Module):
    """
    k-hop Graph Message Passing Layer.
    Propagates local features across neighboring agents using normalized graph adjacency.
    """

    def __init__(self, in_features: int, out_features: int, k_hops: int = 1):
        super().__init__()
        self.k_hops = k_hops
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(in_features if l == 0 else out_features, out_features))
             for l in range(k_hops)]
        )
        for w in self.weights:
            nn.init.xavier_uniform_(w)

    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: (batch_size, num_agents, feat_dim) or (num_agents, feat_dim)
            adj_matrix: (num_agents, num_agents) normalized adjacency matrix
            edge_mask: optional binary or continuous mask (num_agents, num_agents) for BayesG
        """
        is_batched = node_features.dim() == 3
        if not is_batched:
            node_features = node_features.unsqueeze(0)

        A = adj_matrix
        if edge_mask is not None:
            # Gated adjacency by BayesG communication mask
            A = A * edge_mask

        # Ensure normalized
        deg = torch.sum(A, dim=-1, keepdim=True).clamp(min=1e-6)
        A_norm = A / deg

        H = node_features
        for l in range(self.k_hops):
            # Matrix multiplication over graph nodes: A_norm @ H @ W
            # (batch, N, N) @ (batch, N, F_in) -> (batch, N, F_in)
            A_expanded = A_norm.expand(H.size(0), -1, -1)
            H_agg = torch.bmm(A_expanded, H)
            H = F.relu(torch.matmul(H_agg, self.weights[l]))

        if not is_batched:
            H = H.squeeze(0)
        return H
