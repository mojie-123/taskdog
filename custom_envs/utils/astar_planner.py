"""A* grid path planner.

P. Hart et al. "A Formal Basis for the Heuristic Determination of
Minimum Cost Paths." IEEE Trans. SSC, 1968.
"""

import heapq
from math import sqrt


def astar_plan(grid, start, goal, allow_diagonal: bool = True):   # grid：2D 二值地图，0=空闲，1=障碍物。allow_diagonal：是否允许对角线移动（8 连通 vs 4 连通）
    """Plan a path on a 2D binary occupancy grid using A*.

    Args:
        grid:          2D numpy array, 0=free, 1=obstacle.
        start, goal:   (row, col) grid indices.
        allow_diagonal: whether diagonal moves are allowed (8-connectivity).

    Returns:
        List of (row, col) waypoints from start to goal (inclusive),
        or None if no path exists.
    """
    H, W = grid.shape

    # Validate
    for r, c in (start, goal):
        if not (0 <= r < H and 0 <= c < W):   # 检查起点和终点是否在地图范围内
            return None
        if grid[r, c] != 0:   # 检查起点和终点是否为空闲区域（不能是障碍物）
            return None

    # 8-connected neighbours
    if allow_diagonal:
        _neighbours = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, sqrt(2)), (-1, 1, sqrt(2)),
            (1, -1, sqrt(2)), (1, 1, sqrt(2)),
        ]   # 定义邻居方向（走对角线的移动成本是上下左右的1.414倍）
    else:
        _neighbours = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        ]

    # heuristic: octile distance (for 8-connectivity) or Manhattan (for 4)
    if allow_diagonal:    # 启发式函数用Octile距离
        def _h(row, col):
            dr = abs(row - goal[0])
            dc = abs(col - goal[1])
            return sqrt(2) * min(dr, dc) + abs(dr - dc)
    else:   # 启发式函数用曼哈顿距离
        def _h(row, col):
            return abs(row - goal[0]) + abs(col - goal[1])

    open_set = []  # (f, g, row, col)   优先队列，待探索节点，按 f 值排序
    heapq.heappush(open_set, (_h(*start), 0.0, start[0], start[1]))   # 将起点加入 A* 算法的优先队列（open set）
    came_from = {}   # 记录每个节点的前驱节点
    g_score = {start: 0.0}   # 	起点到当前节点的实际成本
    closed = set()   # 已探索的节点

    while open_set:
        f, g, r, c = heapq.heappop(open_set)   # 探索h最小的节点
        if (r, c) in closed:
            continue
        closed.add((r, c))

        if (r, c) == goal:   # 到达目的地，反向获取路径
            return _reconstruct(came_from, (r, c))

        for dr, dc, cost in _neighbours:   # 搜索邻居
            nr, nc = r + dr, c + dc   # 计算邻居坐标 (nr, nc)
            if not (0 <= nr < H and 0 <= nc < W):   # 检查边界和障碍物
                continue
            if grid[nr, nc] != 0:
                continue
            ng = g + cost   # 计算新 g 值 ng = g + cost
            if ng < g_score.get((nr, nc), float("inf")):   # 获取邻居节点之前记录的最小 g 值，如果没记录过则返回无穷大
                g_score[(nr, nc)] = ng   # 记录该邻居点最小g值
                came_from[(nr, nc)] = (r, c)   # 记录邻居的前驱节点（用于最后重建路径）
                heapq.heappush(open_set, (ng + _h(nr, nc), ng, nr, nc))   # 将邻居加入优先队列，优先级是 f = ng + h(nr, nc)

    return None  # no path


def _reconstruct(came_from, current):
    """Reconstruct path from goal to start, then reverse."""
    path = [current]
    while current in came_from:   # 找每个节点的父节点，回溯重建
        current = came_from[current]
        path.append(current)
    path.reverse()   # 反转路径
    return path
