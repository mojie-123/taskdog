"""LiDAR point cloud preprocessing for RL observations.

Raw RayCaster output: [N, B, 3] world coordinates (B ≈ 14,400).
We downsample to K=64 nearest points → flatten to [N, 192].
"""

import torch


def lidar_knn_downsample(
    ray_hits_w: torch.Tensor,
    sensor_pos_w: torch.Tensor,
    num_points: int = 64,
    max_range: float = 70.0,
) -> torch.Tensor:
    """Take K nearest LiDAR hit points and flatten to a 1D observation vector.

    Args:
        ray_hits_w:  [N, B, 3]  world-frame hit coordinates (inf = no hit).
        sensor_pos_w: [N, 3]    world-frame sensor origin per env.
        num_points:   int       number of nearest points to keep.
        max_range:    float     beyond this distance = treat as no-hit.

    Returns:
        [N, num_points * 3]  flattened (x1,y1,z1, x2,y2,z2, …).
    """
    N, B, _ = ray_hits_w.shape

    # 1) Compute distance from sensor to each hit point
    delta = ray_hits_w - sensor_pos_w.unsqueeze(1)   # [N, B, 3]
    dist = torch.linalg.norm(delta, dim=-1)           # [N, B]

    # 2) Replace inf / out-of-range distances with a large sentinel
    dist = torch.where(
        torch.isfinite(dist) & (dist <= max_range),
        dist,
        torch.full_like(dist, max_range * 10.0),
    )

    # 3) Sort by distance and keep the K nearest
    _, idx = torch.topk(dist, k=num_points, dim=-1, largest=False)  # [N, K]
    nearest = torch.gather(
        ray_hits_w, dim=1,
        index=idx.unsqueeze(-1).expand(-1, -1, 3),
    )  # [N, K, 3]

    # 4) Replace coordinates beyond max_range with zeros
    nearest_dist = torch.gather(dist, dim=1, index=idx)  # [N, K]
    mask = (nearest_dist > max_range).unsqueeze(-1)       # [N, K, 1]
    nearest = torch.where(mask, torch.zeros_like(nearest), nearest)

    # 5) Flatten to [N, K*3]
    return nearest.reshape(N, -1)
