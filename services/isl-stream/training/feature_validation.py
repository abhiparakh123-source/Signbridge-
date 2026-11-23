"""Memory-bounded validation helpers for extracted NumPy feature arrays."""
from __future__ import annotations

import numpy as np


def validate_feature_array(array, *, chunk_rows: int = 4096) -> tuple[int, int]:
    """Validate a 2-D feature array without allocating a full-size finite mask."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError("feature array must be a 2-D NumPy array")
    frames, feature_dim = array.shape
    if frames < 1 or feature_dim < 1:
        raise ValueError("feature array must have at least one frame and feature")
    for start in range(0, frames, chunk_rows):
        block = array[start:start + chunk_rows]
        if not np.isfinite(block).all():
            raise ValueError(f"feature array contains non-finite values in rows {start}:{min(start + chunk_rows, frames)}")
    return int(frames), int(feature_dim)
