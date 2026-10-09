"""In-process N2M inference for a live XYZRGB observation."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from custom_envs.scripts.N2M.geometry import transform_se2, wrap_angle


def _pose_rows(relative_poses: np.ndarray, observation_pose_world) -> list[dict]:
    observation_pose_world = np.asarray(
        observation_pose_world, dtype=np.float64
    ).reshape(3)
    rows: list[dict] = []
    for pose in np.asarray(relative_poses, dtype=np.float64).reshape(-1, 3):
        relative = [float(pose[0]), float(pose[1]), wrap_angle(float(pose[2]))]
        rows.append(
            {
                "pose_observation": relative,
                "pose_world": transform_se2(observation_pose_world, relative).tolist(),
            }
        )
    return rows


class N2MRuntime:
    """Load one checkpoint once and predict M poses from live arrays."""

    def __init__(self, checkpoint: Path | str, device: str = "cuda"):
        import torch
        from custom_envs.scripts.N2M.vendor_n2m import N2MNet

        self.torch = torch
        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for N2M but is unavailable")

        payload = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        self.config = payload["config"]
        model_config = copy.deepcopy(self.config["model"])
        # The trained checkpoint already contains the complete PointBERT encoder.
        model_config["encoder"]["ckpt"] = None
        self.model = N2MNet(model_config).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.point_count = int(self.config["dataset"].get("num_points", 8192))

    def close(self) -> None:
        """Release the large encoder before AnyGrasp is loaded on the same GPU."""
        model = getattr(self, "model", None)
        if model is not None:
            model.to("cpu")
            del self.model
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()

    def _point_tensor(self, xyz: np.ndarray, rgb: np.ndarray):
        xyz = np.asarray(xyz, dtype=np.float32)
        rgb = np.asarray(rgb, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
            raise ValueError("xyz and rgb must both have shape (N, 3)")
        if len(xyz) == 0:
            raise ValueError("live point cloud is empty")
        if not np.isfinite(xyz).all() or not np.isfinite(rgb).all():
            raise ValueError("live point cloud contains non-finite values")
        if rgb.size and float(rgb.max()) > 1.0:
            rgb = rgb / 255.0
        rng = np.random.default_rng(0)
        indices = rng.choice(
            len(xyz), self.point_count, replace=len(xyz) < self.point_count
        )
        points = np.concatenate((xyz[indices], rgb[indices]), axis=1)
        return self.torch.from_numpy(points).unsqueeze(0).to(self.device)

    def predict(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        observation_pose_world,
        *,
        observation_id: str = "online_observation",
        num_samples: int = 100,
        sample_seed: int = 42,
    ) -> dict:
        """Return the same prediction schema used by offline evaluation."""
        if int(num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        observation_pose = np.asarray(
            observation_pose_world, dtype=np.float64
        ).reshape(3)
        points = self._point_tensor(xyz, rgb)
        torch = self.torch
        with torch.no_grad():
            means, covariance, weights = self.model(points)
            generator = torch.Generator(device=self.device).manual_seed(int(sample_seed))
            components = torch.multinomial(
                weights[0], int(num_samples), replacement=True, generator=generator
            )
            noise = torch.randn(
                int(num_samples),
                3,
                device=self.device,
                generator=generator,
                dtype=means.dtype,
            )
            selected_means = means[0][components]
            selected_covariance = covariance[0][components]
            cholesky = torch.linalg.cholesky(selected_covariance)
            samples = selected_means + torch.bmm(
                cholesky, noise.unsqueeze(-1)
            ).squeeze(-1)

        means_np = means[0].detach().cpu().numpy()
        samples_np = samples.detach().cpu().numpy()
        return {
            "schema_version": 1,
            "checkpoint": str(self.checkpoint_path),
            "observation_id": str(observation_id),
            "observation_pose_world": observation_pose.tolist(),
            "sample_seed": int(sample_seed),
            "modes": [
                {
                    **pose,
                    "weight": float(weights[0, index].detach().cpu()),
                    "covariance": covariance[0, index].detach().cpu().tolist(),
                }
                for index, pose in enumerate(_pose_rows(means_np, observation_pose))
            ],
            "samples": _pose_rows(samples_np, observation_pose),
        }
