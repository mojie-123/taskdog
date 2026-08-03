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

        # persistent hit counter for mark_elevated (accumulates across calls)
        self._elevated_count = np.zeros((height, width), dtype=np.int32)

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

    def get_inflated_binary_map(self, robot_radius: float = 0.3):
        """Binary obstacle map with obstacles inflated by *robot_radius*.

        This ensures A* keeps the robot centre at least *robot_radius*
        away from any detected obstacle, preventing collisions from the
        robot's physical width.

        Args:
            robot_radius: inflation radius in meters (default 0.3 m).
        """
        from scipy.ndimage import binary_dilation

        binary = self.get_binary_map()
        cells = max(1, int(robot_radius / self.resolution))
        struct = np.ones((2 * cells + 1, 2 * cells + 1), dtype=bool)
        return binary_dilation(binary, structure=struct).astype(np.uint8)

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
        """Bresenham ray: mark cells along the ray as free only.

        The endpoint is NOT marked as occupied here — that is handled
        separately by :meth:`mark_elevated` for above-ground hits.
        """
        sc, sr = self.world_to_grid(sx, sy)
        ec, er = self.world_to_grid(ex, ey)
        sr, sc = max(0, min(sr, self.height-1)), max(0, min(sc, self.width-1))
        er, ec = max(0, min(er, self.height-1)), max(0, min(ec, self.width-1))

        for r, c in self._bresenham(sr, sc, er, ec):
            if 0 <= r < self.height and 0 <= c < self.width:
                self.grid[r, c] -= 0.4   # free
            self.grid[r, c] = float(np.clip(self.grid[r, c], -10.0, 10.0))

    # ------------------------------------------------------------------
    # Elevated-hit projection
    # ------------------------------------------------------------------

    def mark_elevated(self, hits_xy: np.ndarray, min_hits: int = 3):
        """Project elevated LiDAR hits onto the 2D grid as occupied cells.

        Accumulates hits across calls in a persistent counter, so cells only
        become occupied after *min_hits* total hits across the entire session.

        Args:
            hits_xy: (N, 2) numpy array of world (x, y) positions from
                     above-ground LiDAR hits.
            min_hits: minimum number of hits in a cell to confirm occupancy.
        """
        for x, y in hits_xy:
            r, c = self.world_to_grid(x, y)
            if 0 <= r < self.height and 0 <= c < self.width:
                self._elevated_count[r, c] += 1

        occupied_mask = self._elevated_count >= min_hits
        if occupied_mask.any():
            self.grid[occupied_mask] = 10.0

        # Debug: print progress every ~200 total elevated hits
        total = self._elevated_count.sum()
        if total % 200 < len(hits_xy) or total < 10:
            n_occ = (self._elevated_count >= min_hits).sum()
            max_c = self._elevated_count.max()
            print(f"[GRID] elevated total={total}, occupied_cells={n_occ}, "
                  f"max_hits_per_cell={max_c}")

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
