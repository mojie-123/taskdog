"""Semantic point-cloud to ground-query affordance network."""

from __future__ import annotations

import torch
from torch import nn

from .point_encoder import FourierXYZ, MLP, batched_gather, semantic_token_indices


class FloAffNet(nn.Module):
    """Predict one affordance value for every target-centered ground query.

    Inputs
    ------
    scene_points:
        ``[B, N, 5]`` containing XYZ, ``is_target`` and ``is_anchor``.
    query_xyz:
        ``[B, Q, 3]`` canonical ground-query coordinates.

    The network intentionally has no robot-size or explicit-yaw input.  The
    fixed robot morphology is implicit in the labels and the scene has already
    been canonicalized using the frozen AnchorAxisNet prediction.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        token_count: int = 512,
        target_token_fraction: float = 0.25,
        num_heads: int = 4,
        scene_layers: int = 2,
        query_layers: int = 1,
        frequency_bands: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.token_count = int(token_count)
        self.target_token_fraction = float(target_token_fraction)
        self.position_encoding = FourierXYZ(frequency_bands)

        position_dim = self.position_encoding.output_dim
        self.scene_content = MLP(5, hidden_dim, hidden_dim, layers=3, dropout=dropout)
        self.scene_position = MLP(position_dim, hidden_dim, hidden_dim, layers=2)
        scene_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.scene_transformer = nn.TransformerEncoder(
            scene_layer, num_layers=int(scene_layers)
        )

        self.query_content = MLP(position_dim, hidden_dim, hidden_dim, layers=3)
        query_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.query_transformer = nn.TransformerEncoder(
            query_layer, num_layers=int(query_layers)
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.global_condition = MLP(
            2 * hidden_dim, hidden_dim, hidden_dim, layers=2, dropout=dropout
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, scene_points: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
        if scene_points.ndim != 3 or scene_points.shape[-1] != 5:
            raise ValueError(f"expected scene_points [B,N,5], got {scene_points.shape}")
        if query_xyz.ndim != 3 or query_xyz.shape[-1] != 3:
            raise ValueError(f"expected query_xyz [B,Q,3], got {query_xyz.shape}")

        indices = semantic_token_indices(
            scene_points[..., 3:5],
            self.token_count,
            target_fraction=self.target_token_fraction,
            random=self.training,
        )
        tokens = batched_gather(scene_points, indices)
        token_xyz = tokens[..., :3]
        token_features = self.scene_content(tokens)
        token_features = token_features + self.scene_position(
            self.position_encoding(token_xyz)
        )
        token_features = self.scene_transformer(token_features)

        global_feature = torch.cat(
            [token_features.amax(dim=1), token_features.mean(dim=1)], dim=-1
        )
        global_feature = self.global_condition(global_feature).unsqueeze(1)

        queries = self.query_content(self.position_encoding(query_xyz))
        queries = self.query_transformer(queries)
        attended, _ = self.cross_attention(
            query=queries,
            key=token_features,
            value=token_features,
            need_weights=False,
        )
        fused = self.cross_norm(queries + attended + global_feature)
        return self.output_head(fused).squeeze(-1)

    def predict(self, scene_points: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(scene_points, query_xyz))
