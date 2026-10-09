"""Minimal PointBERT encoder adapted from the N2M upstream snapshot."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from .grouping import Group, GroupEncoder


class DropPath(nn.Module):
    """Per-sample stochastic depth, equivalent to the timm layer used upstream."""

    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return value
        keep = 1.0 - self.probability
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        mask = value.new_empty(shape).bernoulli_(keep)
        return value * mask / keep


class MLP(nn.Module):
    def __init__(self, dimension: int, hidden_dimension: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class Attention(nn.Module):
    def __init__(self, dimension: int, heads: int):
        super().__init__()
        if dimension % heads:
            raise ValueError("transformer dimension must be divisible by num_heads")
        self.heads = int(heads)
        self.scale = (dimension // heads) ** -0.5
        self.qkv = nn.Linear(dimension, dimension * 3, bias=False)
        self.projection = nn.Linear(dimension, dimension)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, channels = value.shape
        qkv = self.qkv(value).reshape(
            batch, tokens, 3, self.heads, channels // self.heads
        ).permute(2, 0, 3, 1, 4)
        query, key, item_value = qkv.unbind(0)
        weights = (query @ key.transpose(-2, -1) * self.scale).softmax(dim=-1)
        output = (weights @ item_value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.projection(output), weights


class Block(nn.Module):
    def __init__(self, dimension: int, heads: int, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = Attention(dimension, heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dimension)
        self.mlp = MLP(dimension, dimension * 4)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention, weights = self.attention(self.norm1(value))
        value = value + self.drop_path(attention)
        value = value + self.drop_path(self.mlp(self.norm2(value)))
        return value, weights


class PointTransformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        dimension = int(config["trans_dim"])
        depth = int(config["depth"])
        self.dimension = dimension
        self.group = Group(config["num_group"], config["group_size"])
        self.encoder = GroupEncoder(config["encoder_dims"], input_dim=6)
        self.reduce = nn.Linear(config["encoder_dims"], dimension)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.cls_position = nn.Parameter(torch.randn(1, 1, dimension))
        self.position = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, dimension))
        rates = torch.linspace(0.0, float(config.get("drop_path_rate", 0.0)), depth)
        self.blocks = nn.ModuleList(
            Block(dimension, int(config["num_heads"]), float(rate)) for rate in rates
        )
        self.norm = nn.LayerNorm(dimension)

    @property
    def output_dimension(self) -> int:
        return self.dimension * 2

    @staticmethod
    def _upstream_key_to_local(key: str) -> str:
        """Map the original Point-BERT module names to this vendored encoder."""
        key = key.removeprefix("module.")
        replacements = (
            ("cls_pos", "cls_position"),
            ("reduce_dim", "reduce"),
            ("pos_embed", "position"),
            ("blocks.blocks.", "blocks."),
            (".attn.qkv.", ".attention.qkv."),
            (".attn.proj.", ".attention.projection."),
            (".mlp.fc1.", ".mlp.layers.0."),
            (".mlp.fc2.", ".mlp.layers.3."),
        )
        for source, destination in replacements:
            key = key.replace(source, destination)
        return key

    def load_checkpoint(self, path: str | Path) -> dict:
        """Load an official/N2M Point-BERT checkpoint with hard coverage checks.

        Official ModelNet weights consume XYZ.  This project consumes XYZRGB,
        so the first convolution is expanded from 3 to 6 channels: pretrained
        XYZ weights are retained and RGB weights start at zero.  This preserves
        the exact pretrained function before fine-tuning.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"checkpoint must be a dict, got {type(checkpoint)!r}")
        for wrapper in ("base_model", "model", "model_state_dict", "state_dict"):
            if wrapper in checkpoint and isinstance(checkpoint[wrapper], dict):
                source_state = checkpoint[wrapper]
                break
        else:
            source_state = checkpoint

        target_state = self.state_dict()
        converted = OrderedDict()
        ignored: list[str] = []
        adapted_xyzrgb = False
        shape_errors: list[str] = []
        for source_key, value in source_state.items():
            local_key = self._upstream_key_to_local(source_key)
            if local_key not in target_state:
                ignored.append(source_key)
                continue
            target = target_state[local_key]
            if value.shape != target.shape:
                if (
                    local_key == "encoder.first_conv.0.weight"
                    and value.ndim == 3
                    and value.shape[0] == target.shape[0]
                    and value.shape[1] == 3
                    and target.shape[1] == 6
                    and value.shape[2:] == target.shape[2:]
                ):
                    expanded = torch.zeros_like(target)
                    expanded[:, :3] = value.to(dtype=expanded.dtype)
                    value = expanded
                    adapted_xyzrgb = True
                else:
                    shape_errors.append(
                        f"{source_key} -> {local_key}: {tuple(value.shape)} != "
                        f"{tuple(target.shape)}"
                    )
                    continue
            converted[local_key] = value

        if shape_errors:
            raise RuntimeError("incompatible Point-BERT tensor shapes:\n" + "\n".join(shape_errors))
        missing = sorted(set(target_state) - set(converted))
        loaded_elements = sum(target_state[key].numel() for key in converted)
        total_elements = sum(value.numel() for value in target_state.values())
        coverage = loaded_elements / max(total_elements, 1)
        if missing or coverage < 0.99:
            raise RuntimeError(
                f"Point-BERT checkpoint coverage too low: {coverage:.2%}; "
                f"missing={missing}"
            )
        incompatible = self.load_state_dict(converted, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"strict Point-BERT load unexpectedly failed: {incompatible}")
        report = {
            "path": str(Path(path).resolve()),
            "loaded_tensors": len(converted),
            "total_tensors": len(target_state),
            "element_coverage": coverage,
            "adapted_xyz_to_xyzrgb": adapted_xyzrgb,
            "ignored_tensors": ignored,
        }
        print(
            f"[N2M] PointBERT checkpoint loaded: coverage={coverage:.2%} "
            f"xyzrgb_adapted={adapted_xyzrgb} ignored={len(ignored)}"
        )
        return report

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        groups, centers = self.group(points)
        tokens = self.reduce(self.encoder(groups))
        batch = points.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        value = torch.cat((cls, tokens), dim=1)
        position = torch.cat(
            (self.cls_position.expand(batch, -1, -1), self.position(centers)), dim=1
        )
        final_weights = None
        for block in self.blocks:
            value, final_weights = block(value + position)
        value = self.norm(value)
        features = torch.cat((value[:, 0], value[:, 1:].max(dim=1).values), dim=-1)
        return features, final_weights
