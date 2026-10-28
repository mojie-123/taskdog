"""PointNet-style directed table-edge estimator.

The network receives only the visible anchor cloud, expressed in the robot base
frame and centered at the target.  It predicts the directed outward table-edge
normal in the same frame.  Target position and robot yaw are deliberately not
provided as side channels.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )


class LinearBlock(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        layers: list[nn.Module] = [
            nn.Linear(in_features, out_features, bias=False),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        super().__init__(*layers)


class AnchorAxisNet(nn.Module):
    """Predict a directed unit XY vector from a target-centered anchor cloud."""

    def __init__(
        self,
        point_channels: Sequence[int] = (64, 128, 256),
        head_channels: Sequence[int] = (256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        point_channels = tuple(int(value) for value in point_channels)
        head_channels = tuple(int(value) for value in head_channels)
        if not point_channels or not head_channels:
            raise ValueError("point_channels and head_channels must be non-empty")

        point_layers: list[nn.Module] = []
        previous = 3
        for width in point_channels:
            point_layers.append(ConvBlock(previous, width))
            previous = width
        self.point_encoder = nn.Sequential(*point_layers)

        head_layers: list[nn.Module] = []
        previous = 2 * point_channels[-1]
        for width in head_channels:
            head_layers.append(LinearBlock(previous, width, dropout=dropout))
            previous = width
        head_layers.append(nn.Linear(previous, 2))
        self.head = nn.Sequential(*head_layers)

    def forward(self, anchor_xyz: torch.Tensor) -> torch.Tensor:
        """Return unit vectors with shape ``[B, 2]``.

        ``anchor_xyz`` may be ``[B, N, 3]`` or ``[B, 3, N]``.
        """
        if anchor_xyz.ndim != 3:
            raise ValueError(f"expected a 3D tensor, got {tuple(anchor_xyz.shape)}")
        if anchor_xyz.shape[-1] == 3:
            points = anchor_xyz.transpose(1, 2)
        elif anchor_xyz.shape[1] == 3:
            points = anchor_xyz
        else:
            raise ValueError(f"expected XYZ channels, got {tuple(anchor_xyz.shape)}")

        features = self.point_encoder(points)
        pooled = torch.cat(
            [features.amax(dim=-1), features.mean(dim=-1)], dim=1
        )
        axis = self.head(pooled)
        return F.normalize(axis, p=2, dim=1, eps=1e-8)
