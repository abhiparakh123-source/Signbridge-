"""Inspect Épée keypoint file inventory and sizes without downloading arrays."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_ID = "CLERC-DATA/epee"
EXPECTED_KEYPOINT_FILES = 2400


def inspect(api: Any | None = None, revision: str = "main") -> dict:
    """Read repository metadata only; never fetch any .npy file contents."""
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    files = list(api.list_repo_tree(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=revision,
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
    total_bytes = sum(sizes)
    info = api.dataset_info(DATASET_ID, revision=revision)
    return {
        "dataset": DATASET_ID,
        "revision": revision,
        "commit_hash": getattr(info, "sha", None),
        "expected_keypoint_npy_file_count": EXPECTED_KEYPOINT_FILES,
        "keypoint_npy_file_count": len(npy_files),
        "count_matches_expected": len(npy_files) == EXPECTED_KEYPOINT_FILES,
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / (1024 ** 3), 3),
        "min_file_bytes": min(sizes) if sizes else None,
        "max_file_bytes": max(sizes) if sizes else None,
        "average_file_bytes": round(total_bytes / len(sizes)) if sizes else None,
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
            "The exact 2,400-file corpus count is enforced by the CLI.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    report = inspect(revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["keypoint_npy_file_count"] != EXPECTED_KEYPOINT_FILES:
        raise SystemExit(
            f"Expected {EXPECTED_KEYPOINT_FILES} keypoint NPY files, "
            f"found {report['keypoint_npy_file_count']}"
        )
    return 0


if __name__ == "__main__":
    main()
