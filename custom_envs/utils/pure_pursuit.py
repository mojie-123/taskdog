"""Pure Pursuit path tracker.

R. C. Coulter. "Implementation of the Pure Pursuit Path Tracking
Algorithm." CMU Technical Report, 1992.
"""

import numpy as np
from custom_envs.utils.nav_utils import world_to_body


class PurePursuitController:
    """Geometric path tracker — chases a lookahead point on the path."""

    def __init__(self, lookahead_min: float = 0.5, lookahead_ratio: float = 0.5,
                 target_speed: float = 1.0, max_omega: float = 2.0):
        """
        Args:
            lookahead_min:   minimum lookahead distance (m).
            lookahead_ratio: lookahead = max(min, v * ratio).
            target_speed:    desired cruising speed (m/s).
            max_omega:       max angular velocity rad/s (safety clamp).
        """
        self.lookahead_min = lookahead_min
        self.lookahead_ratio = lookahead_ratio
        self.target_speed = target_speed
        self.max_omega = max_omega

    def compute_velocity(self, path, robot_pos, current_vx=0.0):
        """Compute (vx, omega_z) to follow *path*.

        Args:
            path:       list of (wx, wy) world-frame waypoints.
            robot_pos:  (rx, ry, ryaw) current robot pose.
            current_vx: current forward speed (used for lookahead).

        Returns:
            (vx, omega_z) velocity command.  vy is always 0 for non-holonomic.
        """
        if not path or len(path) < 1:
            return (0.0, 0.0)

        # --- 1. lookahead distance ---
        lookahead = max(self.lookahead_min, current_vx * self.lookahead_ratio)

        # --- 2. find lookahead point on path ---
        target = self._find_lookahead(path, robot_pos, lookahead)

        # --- 3. transform target to body frame ---
        dx_body, dy_body = world_to_body(target[0], target[1], robot_pos)

        # --- 4. curvature → omega ---
        curvature = 2.0 * dy_body / max(lookahead * lookahead, 1e-6)
        # Use target speed (not current_vx which may be 0) so that
        # the robot can turn from a stand-still.
        speed_for_omega = max(current_vx, self.target_speed * 0.3)
        omega = speed_for_omega * curvature
        omega = float(np.clip(omega, -self.max_omega, self.max_omega))

        # --- 5. linear speed: slow down on tight turns ---
        vx = self.target_speed / (1.0 + abs(omega) * 0.5)
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
        best = path[-1]
        for pt in path:
            d = np.hypot(pt[0] - rx, pt[1] - ry)
            if d >= lookahead:
                return pt
            best = pt
        return best
