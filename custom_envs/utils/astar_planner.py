"""A* grid path planner.

P. Hart et al. "A Formal Basis for the Heuristic Determination of
Minimum Cost Paths." IEEE Trans. SSC, 1968.
"""

import heapq
from math import sqrt


def astar_plan(grid, start, goal, allow_diagonal: bool = True):
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
        if not (0 <= r < H and 0 <= c < W):
            return None
        if grid[r, c] != 0:
            return None

    # 8-connected neighbours
    if allow_diagonal:
        _neighbours = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, sqrt(2)), (-1, 1, sqrt(2)),
            (1, -1, sqrt(2)), (1, 1, sqrt(2)),
        ]
    else:
        _neighbours = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        ]

    # heuristic: octile distance (for 8-connectivity) or Manhattan (for 4)
    if allow_diagonal:
        def _h(row, col):
            dr = abs(row - goal[0])
            dc = abs(col - goal[1])
            return sqrt(2) * min(dr, dc) + abs(dr - dc)
    else:
        def _h(row, col):
            return abs(row - goal[0]) + abs(col - goal[1])

    open_set = []  # (f, g, row, col)
    heapq.heappush(open_set, (_h(*start), 0.0, start[0], start[1]))
    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    while open_set:
        f, g, r, c = heapq.heappop(open_set)
        if (r, c) in closed:
            continue
        closed.add((r, c))

        if (r, c) == goal:
            return _reconstruct(came_from, (r, c))

        for dr, dc, cost in _neighbours:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if grid[nr, nc] != 0:
                continue
            ng = g + cost
            if ng < g_score.get((nr, nc), float("inf")):
                g_score[(nr, nc)] = ng
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(open_set, (ng + _h(nr, nc), ng, nr, nc))

    return None  # no path


def _reconstruct(came_from, current):
    """Reconstruct path from goal to start, then reverse."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
