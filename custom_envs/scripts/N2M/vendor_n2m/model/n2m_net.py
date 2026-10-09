"""N2M GMM pose predictor adapted from the upstream ``N2Mnet.py``."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .pointbert import PointTransformer


class N2MNet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        encoder_config = config["encoder"]
        if encoder_config["name"].lower() != "pointbert":
            raise ValueError(f"unsupported encoder: {encoder_config['name']}")
        self.encoder = PointTransformer(encoder_config["config"])
        checkpoint = encoder_config.get("ckpt")
        if checkpoint:
            if not Path(checkpoint).exists():
                raise FileNotFoundError(checkpoint)
            self.encoder.load_checkpoint(checkpoint)
        if encoder_config.get("freeze", False):
            self.encoder.requires_grad_(False)

        decoder_config = config["decoder"]
        self.num_gaussians = int(decoder_config["num_gaussians"])
        self.output_dim = int(decoder_config.get("output_dim", 3))
        if self.output_dim != 3:
            raise ValueError("this project represents manipulation poses as [x,y,yaw]")
        hidden = decoder_config.get("layers", [512, 256])
        dropout = float(decoder_config.get("dropout", 0.1))
        layers: list[nn.Module] = []
        previous = self.encoder.output_dimension
        for dimension in hidden:
            layers.extend((nn.Linear(previous, dimension), nn.ReLU(), nn.Dropout(dropout)))
            previous = dimension
        parameters_per_gaussian = self.output_dim + self.output_dim**2 + 1
        layers.append(nn.Linear(previous, self.num_gaussians * parameters_per_gaussian))
        self.decoder = nn.Sequential(*layers)

    @staticmethod
    def _covariance(raw: torch.Tensor) -> torch.Tensor:
        # matrix_exp/cholesky are substantially more stable in fp32 and are
        # not reliably implemented for fp16 on every CUDA/PyTorch build.
        raw = raw.float()
        symmetric = 0.5 * (raw + raw.transpose(-2, -1))
        return torch.matrix_exp(symmetric)

    def forward(self, point_cloud: torch.Tensor):
        features, attention = self.encoder(point_cloud)
        flat = self.decoder(features)
        batch = flat.shape[0]
        mean_end = self.num_gaussians * self.output_dim
        covariance_end = mean_end + self.num_gaussians * self.output_dim**2
        means = flat[:, :mean_end].reshape(batch, self.num_gaussians, self.output_dim)
        raw_covariance = flat[:, mean_end:covariance_end].reshape(
            batch, self.num_gaussians, self.output_dim, self.output_dim
        )
        covariance = self._covariance(raw_covariance)
        weights = flat[:, covariance_end:].reshape(batch, self.num_gaussians).softmax(dim=-1)
        return means, covariance, weights

    @torch.no_grad()
    def sample(self, point_cloud: torch.Tensor, count: int = 100) -> torch.Tensor:
        means, covariance, weights = self(point_cloud)
        batch_samples = []
        for batch_index in range(means.shape[0]):
            components = torch.multinomial(weights[batch_index], count, replacement=True)
            result = torch.empty(count, self.output_dim, device=means.device)
            for component in range(self.num_gaussians):
                mask = components == component
                if mask.any():
                    distribution = torch.distributions.MultivariateNormal(
                        means[batch_index, component], covariance[batch_index, component]
                    )
                    result[mask] = distribution.sample((int(mask.sum()),))
            batch_samples.append(result)
        return torch.stack(batch_samples)
