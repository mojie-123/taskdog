"""Point grouping layers adapted from upstream PointBERT/N2M."""

from __future__ import annotations

import torch
from torch import nn


def index_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    device = points.device
    batch = points.shape[0]
    view_shape = list(indices.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(indices.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, indices, :]


def farthest_point_sample(xyz: torch.Tensor, count: int) -> torch.Tensor:
    """Pure PyTorch FPS matching the upstream fallback implementation."""
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"FPS expects (B,N,3), got {tuple(xyz.shape)}")
    batch, point_count, _ = xyz.shape
    if count > point_count:
        raise ValueError(f"cannot sample {count} centers from {point_count} points")
    centroids = torch.zeros(batch, count, dtype=torch.long, device=xyz.device)
    distance = torch.full((batch, point_count), 1e10, device=xyz.device)
    farthest = torch.randint(0, point_count, (batch,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(batch, device=xyz.device)
    for index in range(count):
        centroids[:, index] = farthest
        centroid = xyz[batch_indices, farthest].view(batch, 1, 3)
        squared_distance = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, squared_distance)
        farthest = torch.max(distance, dim=-1).indices
    return index_points(xyz, centroids)


def square_distance(source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
    distance = -2.0 * torch.matmul(source, destination.transpose(1, 2))
    distance += torch.sum(source**2, dim=-1).unsqueeze(-1)
    distance += torch.sum(destination**2, dim=-1).unsqueeze(1)
    return distance


def knn_point(count: int, xyz: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    if count > xyz.shape[1]:
        raise ValueError(f"cannot group {count} neighbors from {xyz.shape[1]} points")
    return torch.topk(
        square_distance(centers, xyz), count, dim=-1, largest=False, sorted=False
    ).indices


class Group(nn.Module):
    def __init__(self, num_group: int, group_size: int):
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 3 or points.shape[-1] < 3:
            raise ValueError(f"PointBERT expects (B,N,C>=3), got {tuple(points.shape)}")
        xyz, features = points[..., :3], points[..., 3:]
        batch, point_count, _ = xyz.shape
        centers = farthest_point_sample(xyz, self.num_group)
        indices = knn_point(self.group_size, xyz, centers)
        indices = indices + torch.arange(batch, device=xyz.device).view(-1, 1, 1) * point_count
        indices = indices.reshape(-1)
        neighborhood_xyz = xyz.reshape(batch * point_count, 3)[indices]
        neighborhood_xyz = neighborhood_xyz.reshape(
            batch, self.num_group, self.group_size, 3
        )
        neighborhood_xyz = neighborhood_xyz - centers.unsqueeze(2)
        if features.shape[-1] == 0:
            return neighborhood_xyz, centers
        feature_dim = features.shape[-1]
        neighborhood_features = features.reshape(batch * point_count, feature_dim)[indices]
        neighborhood_features = neighborhood_features.reshape(
            batch, self.num_group, self.group_size, feature_dim
        )
        return torch.cat((neighborhood_xyz, neighborhood_features), dim=-1), centers


class GroupEncoder(nn.Module):
    def __init__(self, encoder_dim: int, input_dim: int = 6):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_dim, 1),
        )

    def forward(self, groups: torch.Tensor) -> torch.Tensor:
        batch, group_count, group_size, channels = groups.shape
        groups = groups.reshape(batch * group_count, group_size, channels)
        features = self.first_conv(groups.transpose(1, 2))
        global_features = features.max(dim=2, keepdim=True).values
        features = torch.cat((global_features.expand(-1, -1, group_size), features), dim=1)
        features = self.second_conv(features).max(dim=2).values
        return features.reshape(batch, group_count, -1)

