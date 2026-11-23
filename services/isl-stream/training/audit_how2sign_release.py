"""Inventory a locally authorized How2Sign release before training.

This tool never downloads data, approves a license, or treats English text as
ASL gloss. It reports candidate annotation files for human review only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov"}
ANNOTATION_SUFFIXES = {".csv", ".tsv"}
GLOSS_COLUMNS = {"gloss", "glosses", "gloss_sequence"}
ID_COLUMNS = {"id", "clip_id", "sentence_name", "video_name"}
SPLIT_COLUMNS = {"split", "set"}
SIGNER_COLUMNS = {"signer", "signer_id", "participant", "participant_id"}
TRANSLATION_COLUMNS = {"text", "translation", "english", "english_translation"}


def inspect_table(path: Path) -> dict:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            headers = {str(name or "").strip().lower() for name in (reader.fieldnames or [])}
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        return {"path": str(path), "readable": False, "error": str(exc)}

    gloss_cols = sorted(headers & GLOSS_COLUMNS)
    translation_cols = sorted(headers & TRANSLATION_COLUMNS)
    required_present = bool(gloss_cols and headers & ID_COLUMNS and headers & SPLIT_COLUMNS)
    nonempty_gloss_rows = 0
    if gloss_cols:
        key = gloss_cols[0]
        nonempty_gloss_rows = sum(1 for row in rows if (row.get(key) or "").strip())

    return {
        "path": str(path),
        "readable": True,
        "rows": len(rows),
        "columns": sorted(headers),
        "candidate_gloss_columns": gloss_cols,
        "translation_columns_not_usable_as_gloss": translation_cols,
        "has_signer_column": bool(headers & SIGNER_COLUMNS),
        "nonempty_gloss_rows": nonempty_gloss_rows,
        "candidate_for_manual_review": required_present and nonempty_gloss_rows > 0,
        "training_approved": False,
        "reason": (
            "Candidate only: verify annotation provenance, gloss convention, signer IDs, "
            "split integrity, rights and expert review before creating a training manifest."
            if required_present and nonempty_gloss_rows > 0
            else "Not a usable gloss candidate; translation-only tables must not be relabeled as gloss."
        ),
    }


def audit_release(root: Path) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"data root is not a directory: {root}")
    tables = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in ANNOTATION_SUFFIXES
    )
    videos = sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
    reports = [inspect_table(path) for path in tables]
    candidates = sum(1 for item in reports if item.get("candidate_for_manual_review"))
    return {
        "root": str(root),
        "video_file_count": videos,
        "annotation_table_count": len(tables),
        "candidate_gloss_table_count": candidates,
        "annotation_tables": reports,
        "training_ready": False,
        "warning": (
            "Inventory only. This does not verify dataset license scope, consent, "
            "annotation accuracy, signer-disjoint splits, or media alignment."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_release(args.data_root)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
