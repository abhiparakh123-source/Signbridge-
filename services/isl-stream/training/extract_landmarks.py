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
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

POSE_COUNT = 33
FACE_COUNT = 468
HAND_COUNT = 21
POSE_WIDTH = 5  # x, y, z, visibility, presence
XYZ_WIDTH = 3
FEATURE_DIM = POSE_COUNT * POSE_WIDTH + FACE_COUNT * XYZ_WIDTH + 2 * HAND_COUNT * XYZ_WIDTH + 4


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


def extract_video(video_path: Path, output_path: Path, max_fps: float) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = max_fps
    stride = max(1, int(round(source_fps / max_fps)))
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
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }


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
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_rows = []
    failures = []
    for row in rows:
        clip_id = row["clip_id"].strip()
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
        safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in clip_id)
        feature_rel = Path("features") / f"{safe_id}.npy"
        feature_path = root / feature_rel
        try:
            stats = extract_video(video_path, feature_path, args.max_fps)
            output_rows.append({
                "clip_id": clip_id,
                "signer_id": row["signer_id"].strip(),
                "sign_language": row["sign_language"].strip().upper(),
                "feature_path": feature_rel.as_posix(),
                "gloss": row["label"].strip().upper().replace(",", " "),
                "consent": "yes",
                "split": row["split"].strip().lower(),
            })
            print(json.dumps({"clip_id": clip_id, **stats}))
        except Exception as exc:
            failures.append({"clip_id": clip_id, "error": str(exc)})
    with output_manifest.open("w", newline="", encoding="utf-8") as handle:
        fields = ["clip_id", "signer_id", "sign_language", "feature_path", "gloss", "consent", "split"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "processed": len(output_rows),
        "failed": len(failures),
        "feature_dim": FEATURE_DIM,
        "output_manifest": str(output_manifest),
        "failures": failures,
        "note": "Landmarks are inputs for research only, not sign translations. Review every label and evaluate with unseen signers.",
    }
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
