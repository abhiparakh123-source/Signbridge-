"""Audit a locally obtained GloSL annotations.csv before any training import.

This is an inventory tool only. It never infers consent, approves rights, or
turns English text into gloss. It reports label-type counts and split leakage
so a human can decide whether rows are eligible for a separately reviewed
training manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED = {"id", "split", "text", "gloss", "gloss_type", "signer"}
VALID_SPLITS = {"train", "dev", "validation", "test"}


def audit_annotations(path: Path) -> dict:
    counts = Counter()
    gloss_types = Counter()
    empty_gloss = 0
    empty_signer = 0
    empty_id = 0
    invalid_split = 0
    duplicate_ids = 0
    signer_splits: dict[str, set[str]] = defaultdict(set)
    video_splits: dict[str, set[str]] = defaultdict(set)
    seen_ids: set[str] = set()
    total = 0
    timestamp_invalid = 0
    fps_invalid = 0
    frame_count_invalid = 0
    missing_video_ref = 0
    timing_columns_available = False
    fps_column_available = False
    frame_count_column_available = False
    video_ref_column_available = False

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = {str(x or "").strip() for x in (reader.fieldnames or [])}
        missing = sorted(REQUIRED - headers)
        timing_columns_available = {"start", "end"}.issubset(headers)
        fps_column_available = "fps" in headers
        frame_count_column_available = "n_frames" in headers
        video_ref_column_available = "video_ref" in headers
        if missing:
            raise ValueError("annotations missing required columns: " + ", ".join(missing))
        for row in reader:
            total += 1
            clip_id = (row.get("id") or "").strip()
            split = (row.get("split") or "").strip().lower()
            signer = (row.get("signer") or "").strip()
            gloss = (row.get("gloss") or "").strip()
            gloss_type = (row.get("gloss_type") or "").strip()
            video_ref = (row.get("video_ref") or "").strip()
            if timing_columns_available:
                try:
                    start, end = float(row.get("start") or ""), float(row.get("end") or "")
                    timestamp_invalid += not (start >= 0 and end > start)
                except (TypeError, ValueError):
                    timestamp_invalid += 1
            if fps_column_available:
                try:
                    fps_invalid += not (float(row.get("fps") or "") > 0)
                except (TypeError, ValueError):
                    fps_invalid += 1
            if frame_count_column_available:
                try:
                    frame_count_invalid += not (int(row.get("n_frames") or "") > 0)
                except (TypeError, ValueError):
                    frame_count_invalid += 1
            if video_ref_column_available:
                missing_video_ref += not bool(video_ref)
            counts[split or "<empty>"] += 1
            gloss_types[gloss_type or "<empty>"] += 1
            empty_id += not bool(clip_id)
            empty_signer += not bool(signer)
            empty_gloss += not bool(gloss)
            invalid_split += split not in VALID_SPLITS
            if clip_id:
                duplicate_ids += clip_id in seen_ids
                seen_ids.add(clip_id)
            if signer and split in VALID_SPLITS:
                signer_splits[signer].add(split)
            if video_ref and split in VALID_SPLITS:
                video_splits[video_ref].add(split)

    signer_leaks = sorted(k for k, v in signer_splits.items() if len(v) > 1)
    video_leaks = sorted(k for k, v in video_splits.items() if len(v) > 1)
    return {
        "file": str(path),
        "rows": total,
        "split_counts": dict(sorted(counts.items())),
        "gloss_type_counts": dict(sorted(gloss_types.items())),
        "empty_gloss_rows": empty_gloss,
        "empty_signer_rows": empty_signer,
        "empty_id_rows": empty_id,
        "invalid_split_rows": invalid_split,
        "duplicate_ids": duplicate_ids,
        "timing_columns_available": timing_columns_available,
        "invalid_timestamp_rows": timestamp_invalid if timing_columns_available else None,
        "fps_column_available": fps_column_available,
        "invalid_fps_rows": fps_invalid if fps_column_available else None,
        "frame_count_column_available": frame_count_column_available,
        "invalid_frame_count_rows": frame_count_invalid if frame_count_column_available else None,
        "video_ref_column_available": video_ref_column_available,
        "missing_video_ref_rows": missing_video_ref if video_ref_column_available else None,
        "signers_in_multiple_splits": len(signer_leaks),
        "videos_in_multiple_splits": len(video_leaks),
        "signer_leak_examples": signer_leaks[:20],
        "video_leak_examples": video_leaks[:20],
        "training_approved": False,
        "warning": (
            "Inventory only. A nonempty gloss field is not proof of a valid ASL gloss. "
            "Verify gloss_type semantics, annotation provenance, upstream consent, "
            "license scope, signer-disjoint evaluation, media alignment and rights "
            "before setting permission/consent fields in a training manifest."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_annotations(args.annotations)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
