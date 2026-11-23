"""Audit an approved ASL corpus manifest before media extraction.

This reports metadata coverage only. It does not inspect video, validate ASL
semantics, establish legal permission, or certify translation quality.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median

from asl_corpus_manifest import REQUIRED, validate_rows


def audit_rows(rows: list[dict[str, str]]) -> dict:
    errors = validate_rows(rows)
    if errors:
        return {"ok": False, "errors": errors, "clips": len(rows)}

    split_counts = Counter()
    split_signers: dict[str, set[str]] = {
        "train": set(), "validation": set(), "test": set()
    }
    transcript_kinds = Counter()
    modalities = Counter()
    durations = []
    datasets = Counter()

    for row in rows:
        split = row["split"].strip().lower()
        split_counts[split] += 1
        split_signers[split].add(row["signer_id"].strip())
        transcript_kinds[row["transcript_kind"].strip().lower()] += 1
        modalities[row["modality"].strip().lower()] += 1
        datasets[row["source_dataset"].strip()] += 1
        durations.append(int(row["end_ms"]) - int(row["start_ms"]))

    return {
        "ok": True,
        "clips": len(rows),
        "unique_signers": len({r["signer_id"].strip() for r in rows}),
        "signers_by_split": {k: len(v) for k, v in split_signers.items()},
        "clips_by_split": {k: split_counts[k] for k in ("train", "validation", "test")},
        "transcript_kinds": dict(sorted(transcript_kinds.items())),
        "modalities": dict(sorted(modalities.items())),
        "source_datasets": dict(sorted(datasets.items())),
        "clip_duration_ms": {
            "minimum": min(durations),
            "median": median(durations),
            "maximum": max(durations),
        },
        "warnings": [
            "Metadata audit only; no video frames or landmarks were inspected.",
            "Gloss annotations still require fluent Deaf ASL review.",
            "Translation transcript coverage is not equivalent to gloss coverage.",
            "Confirm source terms and intended-use permissions independently.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        with args.manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED - set(reader.fieldnames or [])
            if missing:
                report = {"ok": False, "errors": ["Missing columns: " + ", ".join(sorted(missing))]}
            else:
                report = audit_rows(list(reader))
    except OSError as exc:
        report = {"ok": False, "errors": [f"Cannot open manifest: {exc}"]}
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
