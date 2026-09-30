"""2D Occupancy Grid Map with Bresenham ray-casting.

Based on Probabilistic Robotics (Thrun et al., 2005), Ch. 9.
"""

import numpy as np


class OccupancyGrid:
    """Log-odds occupancy grid map.

    Each cell stores log(p_occ / p_free).  0 = unknown, >0 = occupied, <0 = free.
    """

    def __init__(self, width: int, height: int, resolution: float = 0.05):   # 地图的栅格尺寸（单元格数量）以及每个栅格代表的实际距离（米）
        """
        Args:
            width, height: grid dimensions in cells.
            resolution:    meters per cell (default 0.05 → 5 cm).
        """
        if int(width) <= 0 or int(height) <= 0:
            raise ValueError("width and height must be positive")
        if not np.isfinite(resolution) or float(resolution) <= 0.0:
            raise ValueError("resolution must be a positive finite number")
        # 地图的宽、高、分辨率
        self.width = int(width)
        self.height = int(height)
        self.resolution = float(resolution)
        self.grid = np.zeros((self.height, self.width), dtype=np.float32)   # 初始化栅格数组，全0（未知/空闲），越接近1越可能是障碍物
        # World coordinate of the map's lower-left corner.  Array row 0 is
        # therefore the minimum world-Y row and column 0 is the minimum
        # world-X column.  This convention matches nav_msgs/OccupancyGrid and
        # the YAML origin consumed by Nav2 map_server.
        self.origin = (0.0, 0.0)

        # persistent hit counter for mark_elevated (accumulates across calls)
        self._elevated_count = np.zeros((self.height, self.width), dtype=np.int32)   # 记录每个栅格被"高处点"命中的累计次数

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_origin(self, wx: float, wy: float):   # 设置栅格地图原点的世界坐标
        """Set the world-coordinate origin of grid cell (0, 0)."""
        self.origin = (wx, wy)

    def update(self, robot_pose, lidar_hits_w):   # 将一帧激光雷达扫描数据融合到占用栅格地图中。robot_pose: 机器人位姿 (rx, ry, ryaw)，世界坐标系。lidar_hits_w: 形状 [B, 3] 的数组，包含 B 个激光击中点的世界坐标 (x, y, z)，已过滤无穷大值。
        """Integrate one LiDAR scan into the grid.

        Args:
            robot_pose:  (rx, ry, ryaw) in world frame.
            lidar_hits_w: [B, 3] world-frame hit coordinates (inf already filtered).
        """
        rx, ry, _ = robot_pose   # 机器人位置
        for i in range(lidar_hits_w.shape[0]):
            hx, hy = lidar_hits_w[i, 0].item(), lidar_hits_w[i, 1].item()
            if not (np.isfinite(hx) and np.isfinite(hy)):
                continue
            self._ray_cast(rx, ry, hx, hy)   # 记录每一条射线最终打中的栅格

    def integrate_scan(
        self,
        sensor_origin_w: np.ndarray,
        lidar_hits_w: np.ndarray,
        obstacle_min_z: float = 0.15,
        obstacle_max_z: float | None = None,
        min_hits: int = 1,
    ):
        """Integrate a world-frame LiDAR scan into the map.

        Unlike :meth:`update`, this API ray-casts from the actual sensor
        origin instead of the robot root.  It is shared by the IsaacLab and
        MuJoCo mapping entry points.

        Args:
            sensor_origin_w: ``(3,)`` LiDAR origin in world coordinates.
            lidar_hits_w: ``(N, 3)`` world-frame hit coordinates.  Non-finite
                rows are ignored.
            obstacle_min_z: Minimum absolute world Z treated as an obstacle.
            obstacle_max_z: Optional maximum obstacle Z.
            min_hits: Persistent hit count required before a cell is occupied.

        Returns:
            The finite hits that were integrated, shaped ``(M, 3)``.
        """
        origin = np.asarray(sensor_origin_w, dtype=np.float64).reshape(-1)
        hits = np.asarray(lidar_hits_w, dtype=np.float64)
        if origin.shape != (3,):
            raise ValueError(
                f"sensor_origin_w must have shape (3,), got {origin.shape}")
        if not np.isfinite(origin).all():
            raise ValueError("sensor_origin_w must contain only finite values")
        if hits.ndim != 2 or hits.shape[1] != 3:
            raise ValueError(
                f"lidar_hits_w must have shape (N, 3), got {hits.shape}")
        if int(min_hits) <= 0:
            raise ValueError("min_hits must be positive")

        finite = hits[np.isfinite(hits).all(axis=1)]
        for hx, hy, _ in finite:
            self._ray_cast(float(origin[0]), float(origin[1]),
                           float(hx), float(hy))

        elevated_mask = finite[:, 2] > float(obstacle_min_z)
        if obstacle_max_z is not None:
            elevated_mask &= finite[:, 2] <= float(obstacle_max_z)
        elevated = finite[elevated_mask]
        if len(elevated) > 0:
            self.mark_elevated(elevated[:, :2], min_hits=min_hits)
        return finite

    def world_to_grid(self, wx: float, wy: float):   # 将世界坐标转换为栅格地图的行列索引
        """World (m) → grid (row, col).  May return out-of-bounds indices."""
        col = int(np.floor((wx - self.origin[0]) / self.resolution))
        row = int(np.floor((wy - self.origin[1]) / self.resolution))
        return (row, col)

    def grid_to_world(self, row: int, col: int):
        """Grid (row, col) → world (x, y) cell centre."""
        wx = self.origin[0] + (col + 0.5) * self.resolution   # col + 0.5：取栅格中心而非边缘
        wy = self.origin[1] + (row + 0.5) * self.resolution
        return (wx, wy)

    def get_binary_map(self):   # 获取二值地图
        """Return a binary 0/1 obstacle map (1 = occupied, for A* etc.).

        Cells with log-odds > 0 are considered occupied.   使用对数几率表示占用概率：正值：占用概率 > 50%，零值：占用概率 = 50%（完全未知），负值：占用概率 < 50%
        """
        return (self.grid > 0.0).astype(np.uint8)

    def get_inflated_binary_map(self, robot_radius: float = 0.3):   # 考虑机器人尺寸，得到膨胀后的二值地图
        """Binary obstacle map with obstacles inflated by *robot_radius*.

        This ensures A* keeps the robot centre at least *robot_radius*
        away from any detected obstacle, preventing collisions from the
        robot's physical width.

        Args:
            robot_radius: inflation radius in meters (default 0.3 m).
        """
        from scipy.ndimage import binary_dilation   # 使用 SciPy 的二值膨胀函数

        binary = self.get_binary_map()
        cells = max(1, int(robot_radius / self.resolution))   # 地图要膨胀的栅格数量，至少1个
        struct = np.ones((2 * cells + 1, 2 * cells + 1), dtype=bool)
        return binary_dilation(binary, structure=struct).astype(np.uint8)

    def get_visualization(self):   # 生成占用地图的可视化图像，可以用 OpenCV 显示
        """Return an 8-bit BGR image suitable for cv2.imshow.

        black=occupied, white=free, grey=unknown.
        """
        img = np.full((self.height, self.width, 3), 128, dtype=np.uint8)  # grey   所有像素初始值为 128（中等灰色）
        img[self.grid > 0.5] = (0, 0, 0)       # black = occupied   阈值 0.5 表示占用概率较高的区域
        img[self.grid < -0.5] = (255, 255, 255)  # white = free
        return img

    def save(self, path: str):   # 将占用栅格地图及其元数据保存为压缩的 .npz 文件
        """Save grid and metadata as .npz."""
        np.savez_compressed(
            path,
            grid=self.grid,   # 占用概率栅格数据
            origin=np.array(self.origin, dtype=np.float32),   # 地图原点世界坐标 (x, y)
            resolution=self.resolution,   # 栅格分辨率（米/格）
            width=self.width,   # 地图宽度（栅格数）
            height=self.height,   # 地图高度（栅格数）
            elevated_count=self._elevated_count,
        )

    @classmethod
    def load(cls, path: str):   # 从 .npz 文件加载之前保存的占用栅格地图
        """Load grid from a .npz file."""
        data = np.load(path)    # 读取文件数据
        obj = cls(
            width=int(data["width"]),
            height=int(data["height"]),
            resolution=float(data["resolution"]),
        )   # 调用构造函数创建新的 OccupancyGrid 对象（cls 是类方法的第一参数，代表类本身）
        obj.grid = data["grid"]   # 恢复栅格数据
        obj.origin = tuple(data["origin"])   # 恢复原点坐标
        if "elevated_count" in data:
            obj._elevated_count = data["elevated_count"].astype(
                np.int32, copy=True)
        return obj

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ray_cast(self, sx: float, sy: float, ex: float, ey: float):   # 使用 Bresenham 画线算法来标记激光射线经过的栅格为空闲区域
        """Bresenham ray: mark cells along the ray as free only.

        The endpoint is NOT marked as occupied here — that is handled
        separately by :meth:`mark_elevated` for above-ground hits.

        sx, sy：射线起点（机器人位置）世界坐标
        ex, ey：射线终点（激光击中点）世界坐标
        只标记空闲区域，不标记障碍物（障碍物由 mark_elevated 处理）
        """

        clipped = self._clip_segment_to_map(sx, sy, ex, ey)
        if clipped is None:
            return
        sx, sy, ex, ey = clipped

        # world_to_grid() returns (row, col).  Keep that ordering throughout;
        # the previous sc/sr assignment transposed every free-space ray.
        sr, sc = self.world_to_grid(sx, sy)
        er, ec = self.world_to_grid(ex, ey)
        sr = int(np.clip(sr, 0, self.height - 1))
        sc = int(np.clip(sc, 0, self.width - 1))
        er = int(np.clip(er, 0, self.height - 1))
        ec = int(np.clip(ec, 0, self.width - 1))

        for r, c in self._bresenham(sr, sc, er, ec):
            if 0 <= r < self.height and 0 <= c < self.width:
                self.grid[r, c] -= 0.4   # free   每个栅格值减去 0.4（降低占用概率，表示空闲）
                self.grid[r, c] = float(np.clip(self.grid[r, c], -10.0, 10.0))   # 将栅格值限制在 [-10.0, 10.0] 范围内，这使用的是对数几率（log-odds）表示法

    def _clip_segment_to_map(self, sx, sy, ex, ey):
        """Liang-Barsky clip of a world-space segment to map bounds."""
        xmin, ymin = map(float, self.origin)
        xmax = xmin + self.width * self.resolution
        ymax = ymin + self.height * self.resolution
        dx, dy = float(ex - sx), float(ey - sy)
        u0, u1 = 0.0, 1.0

        for p, q in ((-dx, sx - xmin), (dx, xmax - sx),
                     (-dy, sy - ymin), (dy, ymax - sy)):
            if abs(p) < 1e-15:
                if q < 0.0:
                    return None
                continue
            t = q / p
            if p < 0.0:
                if t > u1:
                    return None
                u0 = max(u0, t)
            else:
                if t < u0:
                    return None
                u1 = min(u1, t)

        # Upper bounds are exclusive under floor-based world_to_grid().
        x_hi = np.nextafter(xmax, xmin)
        y_hi = np.nextafter(ymax, ymin)
        csx = float(np.clip(sx + u0 * dx, xmin, x_hi))
        csy = float(np.clip(sy + u0 * dy, ymin, y_hi))
        cex = float(np.clip(sx + u1 * dx, xmin, x_hi))
        cey = float(np.clip(sy + u1 * dy, ymin, y_hi))
        return csx, csy, cex, cey

    # ------------------------------------------------------------------
    # Elevated-hit projection
    # ------------------------------------------------------------------

    def mark_elevated(self, hits_xy: np.ndarray, min_hits: int = 3):   # 将高处的激光击中点投影到 2D 栅格上，标记为占用（障碍物）。
        """Project elevated LiDAR hits onto the 2D grid as occupied cells.

        Accumulates hits across calls in a persistent counter, so cells only
        become occupied after *min_hits* total hits across the entire session.

        Args:
            hits_xy: (N, 2) numpy array of world (x, y) positions from
                     above-ground LiDAR hits.
            min_hits: minimum number of hits in a cell to confirm occupancy.
        """
        if int(min_hits) <= 0:
            raise ValueError("min_hits must be positive")
        for x, y in hits_xy:   # 累积命中计数
            r, c = self.world_to_grid(x, y)
            if 0 <= r < self.height and 0 <= c < self.width:
                self._elevated_count[r, c] += 1

        occupied_mask = self._elevated_count >= min_hits   # 生成占用掩码
        if occupied_mask.any():
            self.grid[occupied_mask] = 10.0   # 将满足条件的栅格值设为 10.0（最大占用概率）

        # Debug: print progress every ~200 total elevated hits
        total = self._elevated_count.sum()
        if total % 200 < len(hits_xy) or total < 10:  # 每累积约 200 次命中打印一次进度
            n_occ = (self._elevated_count >= min_hits).sum()
            max_c = self._elevated_count.max()
            print(f"[GRID] elevated total={total}, occupied_cells={n_occ}, "
                  f"max_hits_per_cell={max_c}")

    @staticmethod
    def _bresenham(r0, c0, r1, c1):   # 逐个生成射线路径上的栅格坐标（使用 yield）
        """Yield (row, col) cells along the line from (r0,c0) to (r1,c1).
        Bresenham 算法的核心思想：用整数运算避免浮点数，高效找出最佳近似直线。"""
        dr = abs(r1 - r0)   # 总距离
        dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1   # 步进
        sc = 1 if c1 > c0 else -1
        if dc > dr:   # 列方向更长（更接近水平线），以列为主方向，逐列步进
            err = dc / 2.0   # 初始误差
            c, r = c0, r0
            while c != c1:
                yield (r, c)   # 输出当前栅格
                err -= dr   # 每次累积误差
                if err < 0:   # 误差超过阈值（说白了就是此时偏离超过0.5个栅格）
                    r += sr   # 在行方向上步进一格
                    err += dc   # 重置误差
                c += sc   # 列步进
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
