"""Bounded-memory landmark windowing for streaming experiments."""
from __future__ import annotations

import numpy as np


def iter_bounded_windows(chunks, max_frames: int, overlap_frames: int = 0):
    """Yield (start_frame, end_frame, window) with bounded rolling context.

    Windows are independent inference contexts. Overlap preserves nearby
    context, but does not itself stitch CTC hypotheses into one transcript.
    """
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1:
        raise ValueError("max_frames must be a positive integer")
    if (
        isinstance(overlap_frames, bool)
        or not isinstance(overlap_frames, int)
        or overlap_frames < 0
        or overlap_frames >= max_frames
    ):
        raise ValueError("overlap_frames must be an integer in [0, max_frames)")

    buffer = None
    buffer_start = 0
    total_received = 0
    last_emitted_end = 0
    feature_dim = None

    for chunk in chunks:
        values = np.asarray(chunk)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("each chunk must be a finite [frames,feature_dim] array")
        if values.shape[0] == 0:
            continue
        if feature_dim is None:
            feature_dim = values.shape[1]
            buffer = np.empty((0, feature_dim), dtype=np.float32)
        if values.shape[1] != feature_dim:
            raise ValueError("feature dimensions changed between chunks")

        values = values.astype(np.float32, copy=False)
        offset = 0
        while offset < len(values):
            capacity = max_frames - len(buffer)
            take = min(capacity, len(values) - offset)
            part = values[offset:offset + take]
            buffer = np.concatenate((buffer, part), axis=0)
            offset += take
            total_received += take

            if len(buffer) == max_frames:
                end = buffer_start + len(buffer)
                yield buffer_start, end, buffer.copy()
                last_emitted_end = end
                keep = min(overlap_frames, len(buffer))
                buffer = buffer[-keep:].copy() if keep else buffer[:0].copy()
                buffer_start = end - keep

    if buffer is not None and total_received > last_emitted_end and len(buffer):
        yield buffer_start, buffer_start + len(buffer), buffer.copy()
