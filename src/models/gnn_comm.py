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
    Includes a local self-projection skip path so agents retain local state awareness
     alongside aggregated neighborhood messages.
    """

    def __init__(self, in_features: int, out_features: int, k_hops: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k_hops = max(1, k_hops)

        self.weights = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(in_features if l == 0 else out_features, out_features)
                )
                for l in range(self.k_hops)
            ]
        )
        self.self_linears = nn.ModuleList(
            [
                nn.Linear(in_features if l == 0 else out_features, out_features)
                for l in range(self.k_hops)
            ]
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
            adj_matrix: (num_agents, num_agents) adjacency matrix
            edge_mask: optional binary or continuous mask (num_agents, num_agents)
                       or (batch_size, num_agents, num_agents) for BayesG
        """
        is_batched = node_features.dim() == 3
        if not is_batched:
            node_features = node_features.unsqueeze(0)

        batch_size, num_agents, _ = node_features.shape

        if adj_matrix.dim() == 2:
            A = adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            A = adj_matrix

        if edge_mask is not None:
            if edge_mask.dim() == 2:
                mask_exp = edge_mask.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                mask_exp = edge_mask
            # Gate neighbor connections while preserving self-loops on the diagonal
            eye = torch.eye(num_agents, device=A.device).unsqueeze(0)
            effective_mask = mask_exp * (1.0 - eye) + eye
            A = A * effective_mask

        # Row-normalize adjacency so message scale is degree-invariant
        deg = torch.sum(A, dim=-1, keepdim=True).clamp(min=1e-6)
        A_norm = A / deg

        H = node_features
        for l in range(self.k_hops):
            # Aggregate neighbor features: (batch, N, N) @ (batch, N, F_in)
            H_agg = torch.bmm(A_norm, H)
            msg_proj = torch.matmul(H_agg, self.weights[l])
            self_proj = self.self_linears[l](H)
            H = F.tanh(msg_proj + self_proj)

        if not is_batched:
            H = H.squeeze(0)
        return H
