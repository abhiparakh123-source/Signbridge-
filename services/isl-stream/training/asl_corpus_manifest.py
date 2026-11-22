"""Validate continuous ASL corpus metadata without downloading or relabeling media.

English transcripts are translation targets, never ASL gloss labels. This
metadata gate intentionally does not feed the existing gloss-CTC trainer.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REQUIRED = {
    "source_dataset", "source_clip_id", "signer_id", "split",
    "start_ms", "end_ms", "transcript", "transcript_kind",
    "modality", "license_id", "permission_status", "attribution",
    "source_url",
}
SPLITS = {"train", "validation", "test"}
TRANSCRIPT_KINDS = {"translation", "gloss"}
MODALITIES = {"rgb", "pose", "depth", "rgb+pose", "rgb+depth", "rgb+pose+depth"}


def validate_rows(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    signer_splits: dict[str, set[str]] = {}
    clip_ids: set[tuple[str, str]] = set()
    usable = 0

    if not rows:
        return ["Manifest contains no clips"]

    for line, row in enumerate(rows, start=2):
        prefix = f"Line {line}: "
        values = {key: (row.get(key) or "").strip() for key in REQUIRED}
        missing = sorted(key for key, value in values.items() if not value)
        if missing:
            errors.append(prefix + "missing required fields: " + ", ".join(missing))
            continue

        dataset, clip, signer = (
            values["source_dataset"], values["source_clip_id"], values["signer_id"]
        )
        key = (dataset, clip)
        if key in clip_ids:
            errors.append(prefix + "duplicate source clip ID within dataset")
        clip_ids.add(key)

        split = values["split"].lower()
        if split not in SPLITS:
            errors.append(prefix + "split must be train, validation, or test")
        else:
            signer_splits.setdefault(signer, set()).add(split)

        try:
            start_ms = int(values["start_ms"])
            end_ms = int(values["end_ms"])
            if start_ms < 0 or end_ms <= start_ms:
                raise ValueError
        except ValueError:
            errors.append(prefix + "start_ms/end_ms must be integers with 0 <= start_ms < end_ms")

        kind = values["transcript_kind"].lower()
        if kind not in TRANSCRIPT_KINDS:
            errors.append(prefix + "transcript_kind must be translation or gloss")
        elif kind == "translation" and not values["transcript"]:
            errors.append(prefix + "translation transcript is empty")

        modality = values["modality"].lower()
        if modality not in MODALITIES:
            errors.append(prefix + "unsupported modality declaration")

        # No media is accepted for training until source rights are explicitly reviewed.
        if values["permission_status"].lower() != "approved":
            errors.append(prefix + "source permission_status must be approved before use")
        if not values["license_id"] or not values["attribution"] or not values["source_url"].startswith(("https://", "http://")):
            errors.append(prefix + "license, attribution and source URL must be recorded")

        if not errors or not any(error.startswith(prefix) for error in errors):
            usable += 1

    for signer, splits in signer_splits.items():
        if len(splits) > 1:
            errors.append(
                f"Signer {signer!r} appears in multiple splits; preserve signer-independent evaluation"
            )
    if usable == 0:
        errors.append("No usable rows remain after metadata validation")
    return errors


def validate_manifest(path: Path) -> list[str]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED - set(reader.fieldnames or [])
            if missing:
                return ["Missing columns: " + ", ".join(sorted(missing))]
            return validate_rows(list(reader))
    except OSError as exc:
        return [f"Cannot open manifest: {exc}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    errors = validate_manifest(args.manifest)
    print(json.dumps({"ok": not errors, "errors": errors,
                      "note": "Metadata validation only; does not validate media, signing labels, license interpretation, or translation quality."}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
