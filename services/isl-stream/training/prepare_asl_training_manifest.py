"""Convert reviewed ASL metadata into the existing landmark extractor input.

This is deliberately strict: English translations are never converted to gloss
labels. Every row must have a reviewed gloss, approved source permission,
explicit consent, and a local media_path inside --data-root.
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
    "source_url", "media_path", "consent",
}


def convert_rows(rows: list[dict[str, str]], data_root: Path) -> list[dict[str, str]]:
    root = data_root.resolve()
    output = []
    seen = set()
    signer_splits: dict[str, set[str]] = {}
    for line, row in enumerate(rows, start=2):
        missing = sorted(key for key in REQUIRED if not (row.get(key) or "").strip())
        if missing:
            raise ValueError(f"line {line}: missing fields: {', '.join(missing)}")
        if row["transcript_kind"].strip().lower() != "gloss":
            raise ValueError(
                f"line {line}: transcript_kind is not gloss; English translations "
                "must not be relabeled as ASL gloss"
            )
        if not row["transcript"].strip():
            raise ValueError(f"line {line}: reviewed gloss transcript must not be empty")
        if not row["source_url"].strip().startswith(("https://", "http://")):
            raise ValueError(f"line {line}: source_url must be an HTTP(S) URL")
        if row["permission_status"].strip().lower() != "approved":
            raise ValueError(f"line {line}: source permission is not approved")
        if row["consent"].strip().lower() not in {"yes", "true", "1"}:
            raise ValueError(f"line {line}: explicit signer consent is required")
        if row["modality"].strip().lower() not in {"rgb", "rgb+pose", "rgb+depth", "rgb+pose+depth"}:
            raise ValueError(f"line {line}: video extraction requires an RGB source modality")
        split = row["split"].strip().lower()
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"line {line}: invalid split")
        signer = row["signer_id"].strip()
        signer_splits.setdefault(signer, set()).add(split)
        dataset, source_clip = row["source_dataset"].strip(), row["source_clip_id"].strip()
        key = (dataset, source_clip)
        if key in seen:
            raise ValueError(f"line {line}: duplicate source clip ID in dataset")
        seen.add(key)
        try:
            start, end = int(row["start_ms"]), int(row["end_ms"])
        except ValueError as exc:
            raise ValueError(f"line {line}: timestamps must be integers") from exc
        if start < 0 or end <= start:
            raise ValueError(f"line {line}: invalid clip timestamps")
        relative = Path(row["media_path"].strip())
        if relative.is_absolute():
            raise ValueError(f"line {line}: media_path must be relative to --data-root")
        media = (root / relative).resolve()
        if not media.is_relative_to(root) or not media.is_file():
            raise ValueError(f"line {line}: media file is missing or outside --data-root")
        # Include dataset in the stable ID so identical source IDs across corpora
        # cannot collide in the feature extractor.
        clip_id = f"{dataset}__{source_clip}"
        output.append({
            "clip_id": clip_id,
            "signer_id": signer,
            "sign_language": "ASL",
            "label": " ".join(row["transcript"].strip().upper().split()),
            "video_path": relative.as_posix(),
            "consent": "yes",
            "split": split,
            "source_dataset": dataset,
            "source_clip_id": source_clip,
            "start_ms": str(start),
            "end_ms": str(end),
            "license_id": row["license_id"].strip(),
            "attribution": row["attribution"].strip(),
            "source_url": row["source_url"].strip(),
        })
    conflicts = [signer for signer, splits in signer_splits.items() if len(splits) > 1]
    if conflicts:
        raise ValueError("signers appear in multiple splits: " + ", ".join(sorted(conflicts)))
    if not output:
        raise ValueError("no ASL clips to convert")
    if {row["split"] for row in output} != {"train", "validation", "test"}:
        raise ValueError("converted data must include train, validation, and test splits")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            parser.error("manifest missing columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    converted = convert_rows(rows, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(converted[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(converted)
    print(json.dumps({"ok": True, "clips": len(converted), "output": str(args.output),
                      "warning": "Gloss labels must be genuinely reviewed ASL gloss annotations; this does not convert translation text or validate sign semantics."}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
