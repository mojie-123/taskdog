"""Pure Pursuit path tracker.

R. C. Coulter. "Implementation of the Pure Pursuit Path Tracking
Algorithm." CMU Technical Report, 1992.
"""

import numpy as np
from custom_envs.utils.nav_utils import world_to_body


class PurePursuitController:
    """Geometric path tracker — chases a lookahead point on the path."""
    # Pure Pursuit 是一种经典的几何路径跟踪算法，核心思想是：在参考路径上找一个"前视点"（lookahead point），然后计算如何让机器人朝向这个点运动。

    def __init__(self, lookahead_min: float = 0.5, lookahead_ratio: float = 0.5,
                 target_speed: float = 1.0, max_omega: float = 2.0):
        """
        Args:
            lookahead_min:   minimum lookahead distance (m).   机器人向前看的最小距离。实际计算时lookahead = max(最小值, 当前速度 × 比例系数)
            lookahead_ratio: lookahead = max(min, v * ratio).   前视距离比例系数
            target_speed:    desired cruising speed (m/s).   目标巡航速度，即在直行时想保持的速度
            max_omega:       max angular velocity rad/s (safety clamp).   最大角速度限制
        """
        self.lookahead_min = lookahead_min
        self.lookahead_ratio = lookahead_ratio
        self.target_speed = target_speed
        self.max_omega = max_omega

    def compute_velocity(self, path, robot_pos, current_vx=0.0):   # 这里path中每个点是世界坐标。robot_pos是机器人当前位姿(rx, ry, ryaw)
        """Compute (vx, omega_z) to follow *path*.

        在参考路径上找一个"前视点" ->
        计算机器人如何运动才能到达这个前视点 ->
        输出线速度 vx 和角速度 omega_z

        Args:
            path:       list of (wx, wy) world-frame waypoints.
            robot_pos:  (rx, ry, ryaw) current robot pose.
            current_vx: current forward speed (used for lookahead).

        Returns:
            (vx, omega_z) velocity command.  vy is always 0 for non-holonomic.
        """
        if not path or len(path) < 1:   # 如果路径为空或只有一个点，无法跟踪，返回零速度
            return (0.0, 0.0)

        # --- 1. lookahead distance ---   计算前视距离
        lookahead = max(self.lookahead_min, current_vx * self.lookahead_ratio)

        # --- 2. find lookahead point on path ---   调用 _find_lookahead 方法，在路径上找到距离机器人 ≥ lookahead 距离的第一个点
        target = self._find_lookahead(path, robot_pos, lookahead)

        # --- 3. transform target to body frame ---   将目标点转换到机器人本体坐标系
        dx_body, dy_body = world_to_body(target[0], target[1], robot_pos)

        # --- 4. curvature → omega ---   计算曲率 → 角速度
        curvature = 2.0 * dy_body / max(lookahead * lookahead, 1e-6)   # 曲率公式：κ = 2 * dy / L²
        # Use target speed (not current_vx which may be 0) so that
        # the robot can turn from a stand-still.
        speed_for_omega = max(current_vx, self.target_speed * 0.3)
        omega = speed_for_omega * curvature   # 角速度公式：ω = v * κ（速度 × 曲率）
        omega = float(np.clip(omega, -self.max_omega, self.max_omega))   # 将角速度限制在 [-max_omega, max_omega] 范围内，防止旋转过快

        # --- 5. linear speed: slow down on tight turns ---
        vx = self.target_speed / (1.0 + abs(omega) * 0.5)   # 转弯越急（omega 越大），速度越慢
        vx = max(0.1, vx)  

        return (vx, omega)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_lookahead(self, path, robot_pos, lookahead):
        """Return the first path point ≥ *lookahead* distance from robot.

        If the entire path is closer than lookahead, return the last point.
        """
        rx, ry, _ = robot_pos
        best = path[-1]   # 在路径上找到距离机器人 ≥ lookahead 距离的第一个点
        for pt in path:
            d = np.hypot(pt[0] - rx, pt[1] - ry)
            if d >= lookahead:   # 如果整条路径都比 lookahead 近，返回路径最后一个点
                return pt
            best = pt
        return best
