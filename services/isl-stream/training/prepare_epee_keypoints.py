"""Prepare CLERC Épée v0.4 keypoints for the SIGNBRIDGE CTC baseline.

This adapter is intentionally separate from the 1,699-D MediaPipe Holistic
extractor. Épée arrays are [frames, 128, 3] and are flattened to 384 features.
It does not download data, infer labels, or grant dataset-use permission.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

FEATURE_DIM = 128 * 3
DATASET_ID = "CLERC-DATA/epee"
DATASET_URL = "https://huggingface.co/datasets/CLERC-DATA/epee"


def clean_gloss(value: str) -> str:
    """Normalize annotation text without translating or inventing glosses."""
    tokens = re.sub(r"\s+", " ", str(value).strip().upper()).split()
    if not tokens:
        raise ValueError("annotation contains an empty gloss")
    return " ".join(tokens)


def split_signers(train: list[str], validation: list[str], test: list[str]) -> dict[str, str]:
    groups = {
        "train": {item.strip() for item in train if item.strip()},
        "validation": {item.strip() for item in validation if item.strip()},
        "test": {item.strip() for item in test if item.strip()},
    }
    if any(not values for values in groups.values()):
        raise ValueError("train, validation, and test signer groups must all be non-empty")
    owners: dict[str, str] = {}
    for split, signers in groups.items():
        for signer in signers:
            if signer in owners:
                raise ValueError(f"signer {signer!r} appears in both {owners[signer]} and {split}")
            owners[signer] = split
    return owners


def read_annotations(path: Path, row: dict[str, str], n_frames: int, fps: float) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clip_id, signer_id = row["clip_id"].strip(), row["signer_id"].strip()
    if payload.get("clip_id") != clip_id or payload.get("signer_id") != signer_id:
        raise ValueError(f"{clip_id}: annotation identity does not match metadata")
    if str(payload.get("phrase_id")) != str(row["phrase_id"]).strip():
        raise ValueError(f"{clip_id}: phrase_id differs between metadata and annotation")
    if int(payload.get("n_frames", -1)) != n_frames or not np.isclose(float(payload.get("fps", 0)), fps):
        raise ValueError(f"{clip_id}: frame count or FPS differs between metadata and annotation")
    if str(payload.get("text_en", "")).strip() != row["text_en"].strip():
        raise ValueError(f"{clip_id}: English prompt differs between metadata and annotation")
    parallel_value = str(row["parallel"]).strip().lower()
    if parallel_value not in {"yes", "no"} or bool(payload.get("parallel")) != (parallel_value == "yes"):
        raise ValueError(f"{clip_id}: parallel flag differs or is invalid")
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{clip_id}: no gloss segments")
    glosses = []
    previous_start = -1.0
    duration = n_frames / fps
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError(f"{clip_id}: malformed gloss segment")
        gloss = clean_gloss(segment.get("gloss", ""))
        try:
            start, end = float(segment["start"]), float(segment["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{clip_id}: segment timestamps must be numeric seconds") from exc
        if not np.isfinite([start, end]).all() or start < 0 or end <= start or end > duration + (1.0 / fps):
            raise ValueError(f"{clip_id}: gloss segment is outside clip duration")
        if start < previous_start:
            raise ValueError(f"{clip_id}: gloss segments are not ordered by start time")
        previous_start = start
        glosses.append(gloss)
    annotation_gloss = " ".join(glosses)
    metadata_gloss = " ".join(clean_gloss(row["gloss_sequence"].replace("|", " ")).split())
    if annotation_gloss != metadata_gloss:
        raise ValueError(f"{clip_id}: gloss_sequence differs between metadata and annotation")
    return annotation_gloss


REQUIRED_PERMISSION_USES = {
    "download_dataset",
    "create_derived_features",
    "train_model",
    "retain_model_checkpoint",
    "deploy_private_non_revenue_family_app",
}


def validate_permission_record(path: Path) -> dict:
    """Require a scoped, explicit written-rights review before any conversion."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read permission record: {exc}") from exc
    if not isinstance(record, dict):
        raise ValueError("permission record must be a JSON object")
    if record.get("status") != "approved":
        raise ValueError("permission record status must be approved; conversion is blocked")
    if record.get("dataset_id") != DATASET_ID:
        raise ValueError(f"permission record dataset_id must be {DATASET_ID}")
    if record.get("license_id") != "CC-BY-NC-SA-4.0":
        raise ValueError("permission record license_id must match the dataset licence")
    if not str(record.get("approved_by", "")).strip():
        raise ValueError("permission record must identify the rights approver")
    if not str(record.get("reviewed_at", "")).strip():
        raise ValueError("permission record must include reviewed_at")
    evidence = str(record.get("written_permission_reference", "")).strip()
    parsed = urlsplit(evidence)
    if not evidence or parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValueError("permission record must reference written permission (HTTP(S) URL)")
    uses = record.get("approved_uses")
    if not isinstance(uses, list):
        raise ValueError("permission record approved_uses must be a list")
    missing = sorted(REQUIRED_PERMISSION_USES - set(uses))
    if missing:
        raise ValueError("permission record does not approve required uses: " + ", ".join(missing))
    return record


def convert_dataset(
    root: Path, output_root: Path, train_signers: list[str],
    validation_signers: list[str], test_signers: list[str],
    permission_record: Path,
) -> dict:
    validate_permission_record(permission_record)
    root, output_root = root.resolve(), output_root.resolve()
    metadata_path = root / "metadata.csv"
    if not metadata_path.is_file():
        raise ValueError("dataset root must contain metadata.csv")
    owner = split_signers(train_signers, validation_signers, test_signers)
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"clip_id", "signer_id", "phrase_id", "fps", "n_frames", "hand_slots", "parallel", "text_en", "gloss_sequence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("metadata.csv missing columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    if not rows:
        raise ValueError("metadata.csv contains no clips")

    output_root.mkdir(parents=True, exist_ok=True)
    feature_dir = output_root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows, seen, signer_counts = [], set(), {}
    failures = []
    for row in rows:
        clip_id, signer = row["clip_id"].strip(), row["signer_id"].strip()
        if signer not in owner:
            continue
        try:
            if not clip_id or clip_id in seen:
                raise ValueError("clip_id is empty or duplicated")
            seen.add(clip_id)
            split = owner[signer]
            phrase = row["phrase_id"].strip()
            if not phrase:
                raise ValueError("phrase_id is required for leakage checks")
            if row["hand_slots"].strip().lower() != "verified":
                raise ValueError("hand_slots must be verified for this baseline")
            fps, n_frames = float(row["fps"]), int(row["n_frames"])
            if not np.isfinite(fps) or fps <= 0 or n_frames < 2:
                raise ValueError("fps and n_frames must be positive")
            annotation_path = root / "annotations" / f"{clip_id}.json"
            keypoint_path = root / "keypoints" / f"{clip_id}.npy"
            if not annotation_path.is_file() or not keypoint_path.is_file():
                raise ValueError("annotation JSON or keypoint NPY is missing")
            gloss = read_annotations(annotation_path, row, n_frames, fps)
            values = np.load(keypoint_path, mmap_mode="r", allow_pickle=False)
            if values.shape != (n_frames, 128, 3):
                raise ValueError(f"expected ({n_frames}, 128, 3), received {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError("keypoints contain NaN or infinity")
            # Preserve all 3 coordinates for all 128 landmarks; no schema guessing.
            flattened = np.asarray(values, dtype=np.float32).reshape(n_frames, FEATURE_DIM)
            if not np.isfinite(flattened).all():
                raise ValueError("float32 conversion produced non-finite features")
            safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", clip_id)
            feature_rel = Path("features") / f"{safe_id}.npy"
            feature_path = output_root / feature_rel
            np.save(feature_path, flattened, allow_pickle=False)
            digest = hashlib.sha256(feature_path.read_bytes()).hexdigest()
            manifest_rows.append({
                "clip_id": clip_id, "signer_id": signer, "sign_language": "ASL",
                "feature_path": feature_rel.as_posix(), "gloss": gloss,
                "consent": "yes", "split": split, "source_dataset": DATASET_ID,
                "source_clip_id": clip_id, "phrase_id": phrase,
                "license_id": "CC-BY-NC-SA-4.0",
                "permission_status": "approved",
                "source_url": DATASET_URL, "feature_dim": str(FEATURE_DIM),
                "feature_sha256": digest,
            })
            signer_counts[signer] = signer_counts.get(signer, 0) + 1
        except Exception as exc:
            failures.append({"clip_id": clip_id, "error": str(exc)})

    if failures:
        raise ValueError(json.dumps({"conversion_failures": failures[:30], "failure_count": len(failures)}, indent=2))
    splits = {row["split"] for row in manifest_rows}
    if splits != {"train", "validation", "test"}:
        raise ValueError("selected signers did not produce all three splits")
    if not manifest_rows:
        raise ValueError("no clips matched the selected signer groups")
    fields = list(manifest_rows[0])
    manifest_path = output_root / "features_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    return {
        "ok": True, "clips": len(manifest_rows), "feature_dim": FEATURE_DIM,
        "signer_counts": signer_counts,
        "split_counts": {name: sum(row["split"] == name for row in manifest_rows)
                         for name in ("train", "validation", "test")},
        "manifest": str(manifest_path),
        "warning": "Conversion only. This does not establish legal permission, train a model, or enable live recognition.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-signers", nargs="+", required=True)
    parser.add_argument("--validation-signers", nargs="+", required=True)
    parser.add_argument("--test-signers", nargs="+", required=True)
    parser.add_argument(
        "--permission-record", type=Path, required=True,
        help="path to written permission/rights review record; its presence is not legal validation",
    )
    args = parser.parse_args()
    try:
        result = convert_dataset(
            args.dataset_root, args.output_root, args.train_signers,
            args.validation_signers, args.test_signers, args.permission_record,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
