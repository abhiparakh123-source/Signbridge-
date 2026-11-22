"""Dependency-free latency and dropped-frame metrics for SIGNBRIDGE benchmarks.

Input records describe one phrase/utterance and use monotonic milliseconds:
captured_at_ms, first_provisional_at_ms, stable_at_ms, translated_at_ms.
Frame counters are optional: captured_frames and dropped_frames.
This tool measures supplied timestamps; it does not run recognition.
"""
from __future__ import annotations

import math
from typing import Any, Iterable


def _number(record: dict[str, Any], key: str, *, required: bool = False) -> float | None:
    value = record.get(key)
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite non-negative number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{key} must be a finite non-negative number") from None
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{key} must be a finite non-negative number")
    return result


def _count(record: dict[str, Any], key: str) -> int:
    """Read an exact non-negative integer frame counter."""
    value = record.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a non-negative integer")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise ValueError(f"{key} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return result


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    # Nearest-rank percentile, deterministic even for small test sets.
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 2)


def summarize_latency(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarise per-phrase latency and frame-loss measurements."""
    provisional: list[float] = []
    stable: list[float] = []
    translated: list[float] = []
    captured_total = dropped_total = 0
    phrase_count = 0

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} must be an object")
        captured = _number(record, "captured_at_ms", required=True)
        stages = (
            ("first_provisional_at_ms", provisional),
            ("stable_at_ms", stable),
            ("translated_at_ms", translated),
        )
        previous = captured
        phrase_has_latency = False
        for key, output in stages:
            timestamp = _number(record, key)
            if timestamp is None:
                continue
            if timestamp < previous:
                raise ValueError(f"record {index}: {key} must not precede the previous stage")
            output.append(timestamp - captured)
            phrase_has_latency = True
            previous = timestamp

        phrase_count += int(phrase_has_latency)
        try:
            captured_total += _count(record, "captured_frames")
            dropped_total += _count(record, "dropped_frames")
        except ValueError as exc:
            raise ValueError(f"record {index}: {exc}") from None

    if dropped_total > captured_total:
        raise ValueError("dropped_frames cannot exceed captured_frames")
    denominator = captured_total
    return {
        "phrases": phrase_count,
        "first_provisional_latency_ms": {
            "p50": _percentile(provisional, 0.50),
            "p95": _percentile(provisional, 0.95),
            "samples": len(provisional),
        },
        "stable_latency_ms": {
            "p50": _percentile(stable, 0.50),
            "p95": _percentile(stable, 0.95),
            "samples": len(stable),
        },
        "translation_latency_ms": {
            "p50": _percentile(translated, 0.50),
            "p95": _percentile(translated, 0.95),
            "samples": len(translated),
        },
        "captured_frames": captured_total,
        "dropped_frames": dropped_total,
        "dropped_frame_percent": round(100 * dropped_total / denominator, 2) if denominator else None,
    }
