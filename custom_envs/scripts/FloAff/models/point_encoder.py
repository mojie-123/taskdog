"""Pure-PyTorch point and position encoding blocks.

These blocks replace the unavailable ``pointnet2_utils.py`` dependency in the
upstream FloAff repository.  They intentionally use standard PyTorch operators
so the same checkpoints can be trained and evaluated without custom CUDA
extensions.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class FourierXYZ(nn.Module):
    def __init__(self, frequency_bands: int = 4):
        super().__init__()
        frequencies = 2.0 ** torch.arange(int(frequency_bands), dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)

    @property
    def output_dim(self) -> int:
        return 3 + 6 * int(self.frequencies.numel())

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        angles = xyz.unsqueeze(-1) * self.frequencies * math.pi
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return torch.cat([xyz, encoded.flatten(start_dim=-2)], dim=-1)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        modules: list[nn.Module] = []
        previous = int(in_dim)
        for _ in range(max(int(layers) - 1, 0)):
            modules.extend(
                [
                    nn.Linear(previous, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                ]
            )
            if dropout > 0.0:
                modules.append(nn.Dropout(dropout))
            previous = hidden_dim
        modules.append(nn.Linear(previous, out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def _even_indices(indices: torch.Tensor, count: int) -> torch.Tensor:
    if indices.numel() == 0:
        raise ValueError("cannot sample an empty point class")
    if indices.numel() >= count:
        positions = torch.linspace(
            0, indices.numel() - 1, count, device=indices.device
        ).round().long()
        return indices[positions]
    repeats = (count + indices.numel() - 1) // indices.numel()
    return indices.repeat(repeats)[:count]


def semantic_token_indices(
    semantics: torch.Tensor,
    token_count: int,
    *,
    target_fraction: float = 0.25,
    random: bool,
) -> torch.Tensor:
    """Select a fixed target/anchor-stratified token set for every batch item."""
    if semantics.ndim != 3 or semantics.shape[-1] != 2:
        raise ValueError(f"expected semantics [B,N,2], got {semantics.shape}")
    batch, points, _ = semantics.shape
    if token_count <= 0 or token_count > points:
        raise ValueError(f"invalid token_count={token_count} for N={points}")
    target_count = max(1, min(token_count - 1, round(token_count * target_fraction)))
    anchor_count = token_count - target_count
    result = []
    for batch_index in range(batch):
        target = torch.nonzero(
            semantics[batch_index, :, 0] > semantics[batch_index, :, 1],
            as_tuple=False,
        ).flatten()
        anchor = torch.nonzero(
            semantics[batch_index, :, 1] >= semantics[batch_index, :, 0],
            as_tuple=False,
        ).flatten()
        if target.numel() == 0 or anchor.numel() == 0:
            all_indices = torch.arange(points, device=semantics.device)
            target = target if target.numel() else all_indices
            anchor = anchor if anchor.numel() else all_indices
        if random:
            target = target[torch.randperm(target.numel(), device=target.device)]
            anchor = anchor[torch.randperm(anchor.numel(), device=anchor.device)]
        selected = torch.cat(
            [_even_indices(target, target_count), _even_indices(anchor, anchor_count)]
        )
        if random:
            selected = selected[torch.randperm(selected.numel(), device=selected.device)]
        result.append(selected)
    return torch.stack(result, dim=0)


def batched_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or indices.ndim != 2:
        raise ValueError("values must be [B,N,C] and indices must be [B,K]")
    expanded = indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
    return torch.gather(values, dim=1, index=expanded)
