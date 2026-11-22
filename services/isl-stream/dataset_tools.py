#!/usr/bin/env python3
"""Validate a private, family-recorded sign-video manifest.

This tool does not upload, decode, or train on video. It checks metadata and
split hygiene before a later training pipeline is added.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

REQUIRED = {"clip_id", "signer_id", "sign_language", "label", "video_path", "consent", "split"}
SIGN_LANGUAGES = {"ISL", "ASL", "BSL"}
SPLITS = {"train", "validation", "test"}


def validate(manifest: Path, root: Path) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    signer_splits: dict[str, set[str]] = {}
    label_counts: dict[str, int] = {}
    try:
        handle = manifest.open(newline="", encoding="utf-8")
    except OSError as exc:
        return [f"Cannot open manifest: {exc}"]

    with handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            return ["Missing columns: " + ", ".join(sorted(missing))]
        for line, row in enumerate(reader, start=2):
            clip_id = (row.get("clip_id") or "").strip()
            signer = (row.get("signer_id") or "").strip()
            language = (row.get("sign_language") or "").strip().upper()
            label = (row.get("label") or "").strip()
            video_path = (row.get("video_path") or "").strip()
            consent = (row.get("consent") or "").strip().lower()
            split = (row.get("split") or "").strip().lower()
            prefix = f"Line {line}: "

            if not clip_id or clip_id in seen_ids:
                errors.append(prefix + "clip_id is empty or duplicated")
            seen_ids.add(clip_id)
            if not signer:
                errors.append(prefix + "signer_id is required (use a private pseudonym)")
            if language not in SIGN_LANGUAGES:
                errors.append(prefix + "sign_language must be ISL, ASL, or BSL")
            if not label:
                errors.append(prefix + "label is required")
            if consent not in {"yes", "true", "1"}:
                errors.append(prefix + "explicit signer consent must be yes")
            if split not in SPLITS:
                errors.append(prefix + "split must be train, validation, or test")
            if not video_path:
                errors.append(prefix + "video_path is required")
            else:
                candidate = (root / video_path).resolve()
                try:
                    candidate.relative_to(root.resolve())
                except ValueError:
                    errors.append(prefix + "video_path must stay inside the dataset folder")
                else:
                    if not candidate.is_file():
                        errors.append(prefix + f"video file not found: {video_path}")
            if signer and split in SPLITS:
                signer_splits.setdefault(signer, set()).add(split)
            if label:
                key = f"{language}:{label}"
                label_counts[key] = label_counts.get(key, 0) + 1

    for signer, splits in signer_splits.items():
        if len(splits) > 1:
            errors.append(
                f"Signer {signer!r} appears in multiple splits ({', '.join(sorted(splits))}); "
                "keep all clips from one signer in exactly one split to prevent leakage."
            )
    if not label_counts:
        errors.append("Manifest contains no usable labelled clips")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    errors = validate(args.manifest, args.root)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    print(json.dumps({"ok": True, "manifest_sha256": digest, "message": "Metadata checks passed. This does not validate sign labels or model accuracy."}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
