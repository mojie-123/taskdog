"""Losses for the two FloAff training stages."""

from __future__ import annotations

import torch


def directed_axis_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = torch.nn.functional.normalize(prediction, dim=-1, eps=1e-8)
    target = torch.nn.functional.normalize(target, dim=-1, eps=1e-8)
    return (1.0 - (prediction * target).sum(dim=-1)).mean()


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mask = mask.bool()
    if not mask.any():
        raise ValueError("masked_mse received an empty support mask")
    return ((prediction - target) ** 2)[mask].mean()


def masked_soft_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    mask = mask.to(dtype=prediction.dtype)
    intersection = (prediction * target * mask).sum(dim=-1)
    denominator = ((prediction.square() + target.square()) * mask).sum(dim=-1)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


def floaff_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    support_mask: torch.Tensor,
    *,
    dice_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = torch.sigmoid(logits)
    mse = masked_mse(prediction, target, support_mask)
    dice = masked_soft_dice(prediction, target, support_mask)
    total = mse + float(dice_weight) * dice
    return total, {"mse": mse, "dice": dice, "prediction": prediction}
