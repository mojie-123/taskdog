"""2D Occupancy Grid Map with Bresenham ray-casting.

Based on Probabilistic Robotics (Thrun et al., 2005), Ch. 9.
"""

import numpy as np
import cv2


class OccupancyGrid:
    """Log-odds occupancy grid map.

    Each cell stores log(p_occ / p_free).  0 = unknown, >0 = occupied, <0 = free.
    """

    def __init__(self, width: int, height: int, resolution: float = 0.05):
        """
        Args:
            width, height: grid dimensions in cells.
            resolution:    meters per cell (default 0.05 → 5 cm).
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid = np.zeros((height, width), dtype=np.float32)
        self.origin = (0.0, 0.0)  # world coordinate of grid[0, 0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_origin(self, wx: float, wy: float):
        """Set the world-coordinate origin of grid cell (0, 0)."""
        self.origin = (wx, wy)

    def update(self, robot_pose, lidar_hits_w):
        """Integrate one LiDAR scan into the grid.

        Args:
            robot_pose:  (rx, ry, ryaw) in world frame.
            lidar_hits_w: [B, 3] world-frame hit coordinates (inf already filtered).
        """
        rx, ry, _ = robot_pose
        for i in range(lidar_hits_w.shape[0]):
            hx, hy = lidar_hits_w[i, 0].item(), lidar_hits_w[i, 1].item()
            if not (np.isfinite(hx) and np.isfinite(hy)):
                continue
            self._ray_cast(rx, ry, hx, hy)

    def world_to_grid(self, wx: float, wy: float):
        """World (m) → grid (row, col).  May return out-of-bounds indices."""
        col = int(round((wx - self.origin[0]) / self.resolution))
        row = int(round((wy - self.origin[1]) / self.resolution))
        return (row, col)

    def grid_to_world(self, row: int, col: int):
        """Grid (row, col) → world (x, y) cell centre."""
        wx = self.origin[0] + (col + 0.5) * self.resolution
        wy = self.origin[1] + (row + 0.5) * self.resolution
        return (wx, wy)

    def get_binary_map(self):
        """Return a binary 0/1 obstacle map (1 = occupied, for A* etc.).

        Cells with log-odds > 0 are considered occupied.
        """
        return (self.grid > 0.0).astype(np.uint8)

    def get_visualization(self):
        """Return an 8-bit BGR image suitable for cv2.imshow.

        black=occupied, white=free, grey=unknown.
        """
        img = np.full((self.height, self.width, 3), 128, dtype=np.uint8)  # grey
        img[self.grid > 0.5] = (0, 0, 0)       # black = occupied
        img[self.grid < -0.5] = (255, 255, 255)  # white = free
        return img

    def save(self, path: str):
        """Save grid and metadata as .npz."""
        np.savez_compressed(
            path,
            grid=self.grid,
            origin=np.array(self.origin, dtype=np.float32),
            resolution=self.resolution,
            width=self.width,
            height=self.height,
        )

    @classmethod
    def load(cls, path: str):
        """Load grid from a .npz file."""
        data = np.load(path)
        obj = cls(
            width=int(data["width"]),
            height=int(data["height"]),
            resolution=float(data["resolution"]),
        )
        obj.grid = data["grid"]
        obj.origin = tuple(data["origin"])
        return obj

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ray_cast(self, sx: float, sy: float, ex: float, ey: float):
        """Bresenham ray: free cells along the ray, occupied at the end."""
        sc, sr = self.world_to_grid(sx, sy)
        ec, er = self.world_to_grid(ex, ey)
        # clamp to grid bounds (ball hits near edge can go out)
        sr, sc = max(0, min(sr, self.height-1)), max(0, min(sc, self.width-1))
        er, ec = max(0, min(er, self.height-1)), max(0, min(ec, self.width-1))

        for r, c in self._bresenham(sr, sc, er, ec):
            if 0 <= r < self.height and 0 <= c < self.width:
                if r == er and c == ec:
                    self.grid[r, c] += 0.8   # occupied  (log-odds increment)
                else:
                    self.grid[r, c] -= 0.4   # free
            # clamp to avoid divergence
            self.grid[r, c] = float(np.clip(self.grid[r, c], -10.0, 10.0))

    @staticmethod
    def _bresenham(r0, c0, r1, c1):
        """Yield (row, col) cells along the line from (r0,c0) to (r1,c1)."""
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        if dc > dr:
            err = dc / 2.0
            c, r = c0, r0
            while c != c1:
                yield (r, c)
                err -= dr
                if err < 0:
                    r += sr
                    err += dc
                c += sc
        else:
            err = dr / 2.0
            c, r = c0, r0
            while r != r1:
                yield (r, c)
                err -= dc
                if err < 0:
                    c += sc
                    err += dr
                r += sr
        yield (r1, c1)
