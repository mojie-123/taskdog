"""Numerically stable GMM NLL for planar pose targets."""

from __future__ import annotations

import math

import torch
from torch import nn


def wrapped_pose_difference(target: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    difference = target.unsqueeze(1) - means
    yaw = torch.atan2(torch.sin(difference[..., 2]), torch.cos(difference[..., 2]))
    return torch.cat((difference[..., :2], yaw.unsqueeze(-1)), dim=-1)


class GMMPoseLoss(nn.Module):
    def __init__(
        self,
        weight_entropy_coefficient: float = 0.0,
        mode_separation_coefficient: float = 0.0,
        separation_scale: float = 0.25,
        jitter: float = 1e-5,
    ):
        super().__init__()
        self.weight_entropy_coefficient = float(weight_entropy_coefficient)
        self.mode_separation_coefficient = float(mode_separation_coefficient)
        self.separation_scale = float(separation_scale)
        self.jitter = float(jitter)

    def forward(self, means, covariance, weights, target):
        dimension = means.shape[-1]
        identity = torch.eye(dimension, device=covariance.device, dtype=covariance.dtype)
        covariance = covariance + self.jitter * identity
        cholesky, info = torch.linalg.cholesky_ex(covariance)
        if torch.any(info != 0):
            raise FloatingPointError("predicted covariance is not positive definite")
        difference = wrapped_pose_difference(target, means)
        solved = torch.cholesky_solve(difference.unsqueeze(-1), cholesky).squeeze(-1)
        mahalanobis = torch.sum(difference * solved, dim=-1)
        log_determinant = 2.0 * torch.log(
            torch.diagonal(cholesky, dim1=-2, dim2=-1)
        ).sum(dim=-1)
        component_log_probability = -0.5 * (
            mahalanobis + log_determinant + dimension * math.log(2.0 * math.pi)
        )
        mixture_log_probability = torch.logsumexp(
            component_log_probability + torch.log(weights.clamp_min(1e-8)), dim=1
        )
        nll = -mixture_log_probability.mean()

        weight_entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=1).mean()
        separation_penalty = means.new_zeros(())
        if means.shape[1] > 1 and self.mode_separation_coefficient:
            pairwise = torch.cdist(means, means)
            upper = torch.triu(
                torch.ones_like(pairwise, dtype=torch.bool), diagonal=1
            )
            separation_penalty = torch.exp(
                -(pairwise[upper] ** 2) / max(self.separation_scale**2, 1e-8)
            ).mean()
        total = (
            nll
            - self.weight_entropy_coefficient * weight_entropy
            + self.mode_separation_coefficient * separation_penalty
        )
        return total, {
            "nll": nll.detach(),
            "weight_entropy": weight_entropy.detach(),
            "separation_penalty": separation_penalty.detach(),
        }

