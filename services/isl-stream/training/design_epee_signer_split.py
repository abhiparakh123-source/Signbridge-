"""Create a deterministic, signer-disjoint CLERC Épée v0.4 split plan.

This is a research split plan, not authorization to train. It reads metadata.csv
only and never downloads video or keypoint payloads.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

# Chosen after inspecting the v0.4 metadata's actual signer/phrase assignments.
# Keep the five parallel-grid signers divided 3/1/1 and pair each held-out
# parallel signer with one of the three non-parallel prompt-set signers.
SIGNER_SPLITS = {
    "train": ["ALPHA", "BRAVO", "CHARLIE", "FOXTROT"],
    "validation": ["DELTA", "GOLF"],
    "test": ["ECHO", "HOTEL"],
}


def build_plan(metadata_path: Path) -> dict:
    with metadata_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("metadata.csv is empty")
    required = {"clip_id", "signer_id", "phrase_id", "parallel"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")

    signer_owner = {}
    for split, signers in SIGNER_SPLITS.items():
        for signer in signers:
            if signer in signer_owner:
                raise ValueError(f"signer assigned to multiple splits: {signer}")
            signer_owner[signer] = split

    seen_clips = set()
    split_clips = defaultdict(list)
    split_signers = defaultdict(set)
    split_parallel_phrases = defaultdict(set)
    split_nonparallel_phrases = defaultdict(set)
    for row in rows:
        clip = row["clip_id"].strip()
        signer = row["signer_id"].strip()
        phrase = row["phrase_id"].strip()
        if not clip or clip in seen_clips:
            raise ValueError(f"empty or duplicate clip_id: {clip!r}")
        seen_clips.add(clip)
        if signer not in signer_owner:
            raise ValueError(f"unassigned signer in metadata: {signer}")
        split = signer_owner[signer]
        split_clips[split].append(clip)
        split_signers[split].add(signer)
        target = split_parallel_phrases if row["parallel"].strip().lower() == "yes" else split_nonparallel_phrases
        target[split].add(phrase)

    observed_signers = {r["signer_id"].strip() for r in rows}
    if observed_signers != set(signer_owner):
        raise ValueError(
            f"signer mismatch; missing={sorted(set(signer_owner)-observed_signers)}, "
            f"unexpected={sorted(observed_signers-set(signer_owner))}"
        )
    if len(rows) != 2400 or any(len(split_clips[s]) != n for s, n in {
        "train": 1200, "validation": 600, "test": 600
    }.items()):
        raise ValueError("unexpected v0.4 or split row counts; refusing to emit plan")

    # Explicitly assert no signer crosses split boundaries.
    owners = {s: split for split, members in split_signers.items() for s in members}
    if len(owners) != len(observed_signers):
        raise ValueError("signer leakage detected")

    parallel_intersections = {}
    names = list(SIGNER_SPLITS)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = sorted(split_parallel_phrases[left] & split_parallel_phrases[right])
            parallel_intersections[f"{left}__{right}"] = overlap

    return {
        "dataset": "CLERC-DATA/epee",
        "release": "v0.4",
        "status": "proposed_split_plan_not_training_authorization",
        "strategy": "signer_disjoint_4_2_2; 3_1_1_parallel_grid_signers",
        "signer_splits": SIGNER_SPLITS,
        "split_summary": {
            split: {
                "clips": len(split_clips[split]),
                "signers": sorted(split_signers[split]),
                "parallel_phrase_count": len(split_parallel_phrases[split]),
                "nonparallel_phrase_count": len(split_nonparallel_phrases[split]),
                "clip_ids": sorted(split_clips[split]),
            }
            for split in names
        },
        "parallel_phrase_intersections": parallel_intersections,
        "limitations": [
            "Signer-disjoint, but not phrase-disjoint: the parallel phrase grid intentionally overlaps across splits.",
            "Validation and test each contain only one parallel-grid signer; held-out signer estimates will have high uncertainty.",
            "The ECHO-only parallel phrases are not represented in train or validation.",
            "Dataset license and written permission must be cleared before training or retaining derived model artifacts.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(args.metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in plan.items() if k != "split_summary"}, indent=2))
    print("split_summary:", {
        k: {field: value for field, value in v.items() if field != "clip_ids"}
        for k, v in plan["split_summary"].items()
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
