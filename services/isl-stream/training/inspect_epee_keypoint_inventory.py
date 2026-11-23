"""Inspect Épée keypoint file inventory and sizes without downloading arrays."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi

DATASET_ID = "CLERC-DATA/epee"


def inspect() -> dict:
    api = HfApi()
    files = list(api.list_repo_tree(
        repo_id=DATASET_ID,
        repo_type="dataset",
        path_in_repo="keypoints",
        recursive=True,
        expand=True,
    ))
    npy_files = [
        item for item in files
        if getattr(item, "path", "").endswith(".npy")
        and getattr(item, "size", None) is not None
    ]
    sizes = [int(item.size) for item in npy_files]
    return {
        "dataset": DATASET_ID,
        "keypoint_npy_file_count": len(npy_files),
        "total_bytes": sum(sizes),
        "total_gib": round(sum(sizes) / (1024 ** 3), 3),
        "min_file_bytes": min(sizes) if sizes else None,
        "max_file_bytes": max(sizes) if sizes else None,
        "average_file_bytes": round(sum(sizes) / len(sizes)) if sizes else None,
        "size_buckets": dict(sorted(Counter(
            "under_1MiB" if size < 1024**2 else
            "1_to_10MiB" if size < 10*1024**2 else
            "10_to_50MiB" if size < 50*1024**2 else
            "over_50MiB"
            for size in sizes
        ).items())),
        "downloaded_payload_bytes": 0,
        "notes": [
            "Only Hugging Face repository file metadata was inspected.",
            "No keypoint NPY array payloads or videos were downloaded.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["keypoint_npy_file_count"] == 0:
        raise SystemExit("No keypoint NPY files with known sizes were found")
    return 0


if __name__ == "__main__":
    main()
