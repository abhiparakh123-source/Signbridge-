"""Download only the audited Épée metadata, annotations and keypoint arrays.

This deliberately excludes RGB videos. It requires the same explicit rights
record as the converter and refuses unexpected corpus sizes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

try:
    from .prepare_epee_keypoints import DATASET_ID, validate_permission_record
except ImportError:
    from prepare_epee_keypoints import DATASET_ID, validate_permission_record

EXPECTED_KEYPOINT_FILES = 2400
DEFAULT_MAX_BYTES = 1024 ** 3


def download_keypoints(
    destination: Path,
    permission_record: Path | None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    revision: str = "main",
    user_attested: bool = False,
) -> dict:
    # Validate written evidence or record the user's explicit approval attestation.
    authorization = validate_permission_record(permission_record, user_attested=user_attested)
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    local_path = Path(snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=str(destination),
        allow_patterns=[
            "metadata.csv",
            "annotations/*.json",
            "keypoints/*.npy",
        ],
    )).resolve()
    if not local_path.is_relative_to(destination):
        raise ValueError("downloaded dataset path escaped the requested destination")

    keypoint_files = sorted((local_path / "keypoints").glob("*.npy"))
    if len(keypoint_files) != EXPECTED_KEYPOINT_FILES:
        raise ValueError(
            f"expected {EXPECTED_KEYPOINT_FILES} keypoint arrays, found {len(keypoint_files)}"
        )
    total_bytes = sum(path.stat().st_size for path in keypoint_files)
    if total_bytes > max_bytes:
        raise ValueError(
            f"keypoint payload is {total_bytes} bytes, exceeding limit {max_bytes}"
        )
    if not (local_path / "metadata.csv").is_file():
        raise ValueError("metadata.csv missing after dataset download")
    annotation_files = list((local_path / "annotations").glob("*.json"))
    if len(annotation_files) != EXPECTED_KEYPOINT_FILES:
        raise ValueError(
            f"expected {EXPECTED_KEYPOINT_FILES} annotation JSON files, found {len(annotation_files)}"
        )

    info = HfApi().dataset_info(DATASET_ID, revision=revision)
    return {
        "ok": True,
        "dataset": DATASET_ID,
        "revision": revision,
        "commit_hash": getattr(info, "sha", None),
        "destination": str(local_path),
        "keypoint_files": len(keypoint_files),
        "annotation_files": len(annotation_files),
        "keypoint_bytes": total_bytes,
        "keypoint_gib": round(total_bytes / (1024 ** 3), 3),
        "downloaded_patterns": ["metadata.csv", "annotations/*.json", "keypoints/*.npy"],
        "videos_downloaded": False,
        "authorization_basis": authorization.get("authorization_basis", "written_permission_record"),
        "warning": "Dataset payload downloaded only; conversion, model training and deployment are separate steps.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    authorization_group = parser.add_mutually_exclusive_group(required=True)
    authorization_group.add_argument("--permission-record", type=Path)
    authorization_group.add_argument(
        "--user-attests-rights", action="store_true",
        help="record the user's explicit approval confirmation without fabricating written evidence",
    )
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = download_keypoints(
        args.destination, args.permission_record, args.max_bytes, args.revision,
        user_attested=args.user_attests_rights,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
