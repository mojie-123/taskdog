"""Sampling helpers without a SciPy dependency."""

from __future__ import annotations

import numpy as np


def latin_hypercube(count: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    """Return a randomized Latin hypercube in the open unit cube."""
    count = int(count)
    dimensions = int(dimensions)
    if count <= 0 or dimensions <= 0:
        raise ValueError("count and dimensions must be positive")
    result = np.empty((count, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        values = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
        rng.shuffle(values)
        result[:, dimension] = values
    return result


def scale_unit(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    lower = float(lower)
    upper = float(upper)
    if not lower < upper:
        raise ValueError(f"invalid interval [{lower}, {upper}]")
    return lower + np.asarray(values, dtype=np.float64) * (upper - lower)
