"""Extract fixed-width MediaPipe Holistic landmark sequences from consented clips.

Input manifest uses the family dataset schema. Its label column must contain
expert-reviewed gloss tokens (space-separated), not an English/Hindi sentence.
Output is a separate training manifest and private .npy feature files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

POSE_COUNT = 33
FACE_COUNT = 468
HAND_COUNT = 21
POSE_WIDTH = 5  # x, y, z, visibility, presence
XYZ_WIDTH = 3
FEATURE_DIM = POSE_COUNT * POSE_WIDTH + FACE_COUNT * XYZ_WIDTH + 2 * HAND_COUNT * XYZ_WIDTH + 4


def sampling_stride(source_fps: float, max_fps: float) -> int:
    """Choose a stride that never intentionally samples faster than max_fps."""
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source_fps must be finite and positive")
    if not np.isfinite(max_fps) or max_fps <= 0:
        raise ValueError("max_fps must be finite and positive")
    return max(1, int(math.ceil(source_fps / max_fps)))


def safe_clip_filename(clip_id: str) -> str:
    """Make a deterministic filename and reject IDs with no safe characters."""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in clip_id.strip())
    safe = safe.strip("_")
    if not safe:
        raise ValueError("clip_id must contain at least one letter or digit")
    return safe


def _landmark_values(container, count: int, width: int, fields: tuple[str, ...]) -> tuple[np.ndarray, bool]:
    output = np.zeros((count, width), dtype=np.float32)
    landmarks = getattr(container, "landmark", None) if container is not None else None
    if not landmarks:
        return output.reshape(-1), False
    if len(landmarks) != count:
        raise ValueError(f"expected {count} landmarks, received {len(landmarks)}")
    for index, landmark in enumerate(landmarks):
        for column, field in enumerate(fields):
            value = getattr(landmark, field, 0.0)
            output[index, column] = float(value or 0.0)
    return output.reshape(-1), True


def pack_landmarks(results) -> np.ndarray:
    """Return 1,699 float32 features: pose, face, hands, then four presence flags."""
    pose, pose_present = _landmark_values(
        results.pose_landmarks, POSE_COUNT, POSE_WIDTH,
        ("x", "y", "z", "visibility", "presence"),
    )
    face, face_present = _landmark_values(
        results.face_landmarks, FACE_COUNT, XYZ_WIDTH, ("x", "y", "z")
    )
    left, left_present = _landmark_values(
        results.left_hand_landmarks, HAND_COUNT, XYZ_WIDTH, ("x", "y", "z")
    )
    right, right_present = _landmark_values(
        results.right_hand_landmarks, HAND_COUNT, XYZ_WIDTH, ("x", "y", "z")
    )
    flags = np.asarray(
        [pose_present, face_present, left_present, right_present], dtype=np.float32
    )
    vector = np.concatenate((pose, face, left, right, flags))
    if vector.shape != (FEATURE_DIM,) or not np.isfinite(vector).all():
        raise ValueError("invalid landmark feature vector")
    return vector


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a potentially large feature file without loading it all into memory."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def extract_video(video_path: Path, output_path: Path, max_fps: float) -> dict:
    # Heavy video dependencies are imported only for actual extraction so the
    # feature-vector unit tests can run with NumPy alone.
    import cv2
    import mediapipe as mp

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = max_fps
    stride = sampling_stride(float(source_fps), max_fps)
    frames = []
    frame_index = 0
    try:
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % stride == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    result = holistic.process(rgb)
                    frames.append(pack_landmarks(result))
                frame_index += 1
    finally:
        capture.release()
    if len(frames) < 2:
        raise ValueError(f"clip yielded fewer than two sampled frames: {video_path}")
    sequence = np.stack(frames).astype(np.float32)
    if not np.any(sequence[:, -4:]):
        raise ValueError(f"no pose, face or hand landmarks detected in clip: {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, sequence, allow_pickle=False)
    return {
        "frames": int(sequence.shape[0]),
        "feature_dim": int(sequence.shape[1]),
        "source_fps": float(source_fps),
        "sampled_fps_approx": float(source_fps / stride),
        "sha256": sha256_file(output_path),
    }


def build_feature_manifest_row(
    row: dict[str, str], feature_rel: Path, stats: dict
) -> dict[str, str]:
    """Preserve optional source provenance alongside each extracted feature file."""
    return {
        "clip_id": row["clip_id"].strip(),
        "signer_id": row["signer_id"].strip(),
        "sign_language": row["sign_language"].strip().upper(),
        "feature_path": feature_rel.as_posix(),
        "gloss": row["label"].strip().upper().replace(",", " "),
        "consent": "yes",
        "split": row["split"].strip().lower(),
        "source_dataset": (row.get("source_dataset") or "").strip(),
        "source_clip_id": (row.get("source_clip_id") or row["clip_id"]).strip(),
        "source_media_path": (row.get("video_path") or "").strip(),
        "start_ms": (row.get("start_ms") or "").strip(),
        "end_ms": (row.get("end_ms") or "").strip(),
        "license_id": (row.get("license_id") or "").strip(),
        "attribution": (row.get("attribution") or "").strip(),
        "source_url": (row.get("source_url") or "").strip(),
        "feature_sha256": stats["sha256"],
    }


def validate_signer_splits(rows: list[dict[str, str]]) -> None:
    """Fail before expensive extraction if a signer leaks across data splits."""
    signer_splits: dict[str, set[str]] = {}
    for line, row in enumerate(rows, start=2):
        signer = (row.get("signer_id") or "").strip()
        split = (row.get("split") or "").strip().lower()
        if not signer:
            raise ValueError(f"line {line}: signer_id is required")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"line {line}: split must be train, validation, or test")
        signer_splits.setdefault(signer, set()).add(split)
    conflicts = {
        signer: sorted(splits)
        for signer, splits in signer_splits.items()
        if len(splits) > 1
    }
    if conflicts:
        detail = "; ".join(
            f"{signer}: {', '.join(splits)}"
            for signer, splits in sorted(conflicts.items())
        )
        raise ValueError("signers must be exclusive to one split: " + detail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="family video manifest")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, default=Path("features_manifest.csv"))
    parser.add_argument("--max-fps", type=float, default=60.0)
    args = parser.parse_args()
    if not 1 <= args.max_fps <= 120:
        parser.error("--max-fps must be between 1 and 120")
    root = args.data_root.resolve()
    manifest = args.manifest.resolve()
    output_manifest = args.output_manifest.resolve()
    required = {"clip_id", "signer_id", "sign_language", "label", "video_path", "consent", "split"}
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            parser.error("manifest missing columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    if not rows:
        parser.error("manifest has no clips")
    try:
        validate_signer_splits(rows)
    except ValueError as exc:
        parser.error(str(exc))
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_rows = []
    failures = []
    seen_clip_ids = set()
    seen_output_names = set()
    for row in rows:
        clip_id = row["clip_id"].strip()
        if not clip_id or clip_id in seen_clip_ids:
            failures.append({"clip_id": clip_id, "error": "clip_id is empty or duplicated"})
            continue
        seen_clip_ids.add(clip_id)
        consent = row["consent"].strip().lower()
        if consent not in {"yes", "true", "1"}:
            failures.append({"clip_id": clip_id, "error": "explicit consent is required"})
            continue
        if row["sign_language"].strip().upper() not in {"ISL", "ASL", "BSL"}:
            failures.append({"clip_id": clip_id, "error": "unsupported sign language"})
            continue
        if row["split"].strip().lower() not in {"train", "validation", "test"}:
            failures.append({"clip_id": clip_id, "error": "invalid split"})
            continue
        if not row["label"].strip():
            failures.append({"clip_id": clip_id, "error": "expert-reviewed gloss label is required"})
            continue
        video_path = (root / row["video_path"]).resolve()
        if not video_path.is_relative_to(root) or not video_path.is_file():
            failures.append({"clip_id": clip_id, "error": "video path missing or outside dataset root"})
            continue
        try:
            safe_id = safe_clip_filename(clip_id)
        except ValueError as exc:
            failures.append({"clip_id": clip_id, "error": str(exc)})
            continue
        if safe_id in seen_output_names:
            failures.append({"clip_id": clip_id, "error": "clip_id collides with another sanitized output filename"})
            continue
        seen_output_names.add(safe_id)
        feature_rel = Path("features") / f"{safe_id}.npy"
        feature_path = root / feature_rel
        try:
            stats = extract_video(video_path, feature_path, args.max_fps)
            output_rows.append(build_feature_manifest_row(row, feature_rel, stats))
            print(json.dumps({"clip_id": clip_id, **stats}))
        except Exception as exc:
            failures.append({"clip_id": clip_id, "error": str(exc)})
    with output_manifest.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "clip_id", "signer_id", "sign_language", "feature_path", "gloss",
            "consent", "split", "source_dataset", "source_clip_id",
            "source_media_path", "start_ms", "end_ms", "license_id",
            "attribution", "source_url", "feature_sha256",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "processed": len(output_rows),
        "failed": len(failures),
        "feature_dim": FEATURE_DIM,
        "output_manifest": str(output_manifest),
        "failures": failures,
        "max_fps": args.max_fps,
        "note": "Landmarks are inputs for research only, not sign translations. Review every label and evaluate with unseen signers. Sampling uses a ceiling stride and may be lower than the requested maximum.",
    }
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
