"""Metadata/annotation audit for CLERC Épée v0.4; never downloads video or NPY payloads."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://huggingface.co/datasets/CLERC-DATA/epee/resolve/main"
REQUIRED = {"clip_id", "signer_id", "phrase_id", "parallel", "text_en",
            "fps", "n_frames", "oval_detection", "hand_slots", "gloss_sequence"}


def fetch(url: str, byte_range: str | None = None, limit: int = 2_000_000) -> bytes:
    headers = {"User-Agent": "SIGNBRIDGE-metadata-audit/1.0"}
    if byte_range:
        headers["Range"] = byte_range
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError(f"response exceeded safety limit ({limit} bytes): {url}")
        if byte_range and response.status != 206:
            raise ValueError(f"server did not honor byte range for {url}")
        return data


def npy_shape_from_header(data: bytes) -> tuple[int, ...]:
    import numpy as np
    stream = io.BytesIO(data)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, _ = np.lib.format.read_array_header_1_0(stream)
    elif version in {(2, 0), (3, 0)}:
        shape, _, _ = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"unsupported NPY version: {version}")
    return tuple(shape)


def audit(root: Path | None, metadata_bytes: bytes, check_remote_headers: bool = False) -> dict:
    rows = list(csv.DictReader(io.StringIO(metadata_bytes.decode("utf-8-sig"))))
    report = {
        "dataset": "CLERC-DATA/epee", "release": "v0.4",
        "metadata_rows": len(rows), "signer_counts": dict(sorted(Counter(r.get("signer_id", "") for r in rows).items())),
        "parallel_rows": 0, "parallel_phrase_coverage": {},
        "hand_slot_counts": dict(sorted(Counter(r.get("hand_slots", "") for r in rows).items())),
        "errors": [], "warnings": [], "keypoint_header_checks": 0,
        "keypoint_shape_counts": {}, "video_downloaded": False, "array_payload_downloaded": False,
    }
    if not rows:
        report["errors"].append("metadata.csv is empty")
        return report
    missing = REQUIRED - set(rows[0])
    if missing:
        report["errors"].append("metadata.csv missing columns: " + ", ".join(sorted(missing)))
        return report

    ids, phrase_signers = set(), defaultdict(set)
    signer_phrase_ids = defaultdict(lambda: {"parallel": set(), "nonparallel": set()})
    signer_row_counts = defaultdict(lambda: {"parallel": 0, "nonparallel": 0})
    gloss_count = 0
    oval_scores = []
    shape_counts = Counter()
    remote_candidates = []
    for index, row in enumerate(rows, 2):
        clip = row["clip_id"].strip()
        signer = row["signer_id"].strip()
        try:
            if not clip or clip in ids:
                raise ValueError("empty or duplicate clip_id")
            ids.add(clip)
            if not signer or not row["phrase_id"].strip():
                raise ValueError("missing signer_id or phrase_id")
            fps, frames = float(row["fps"]), int(row["n_frames"])
            if not math.isfinite(fps) or fps <= 0 or frames < 2:
                raise ValueError("invalid fps/n_frames")
            oval = float(row["oval_detection"])
            if not math.isfinite(oval) or not 0 <= oval <= 1:
                raise ValueError("oval_detection outside [0,1]")
            oval_scores.append(oval)
            if row["parallel"].strip().lower() not in {"yes", "no"}:
                raise ValueError("parallel must be yes/no")
            if not row["text_en"].strip() or not row["gloss_sequence"].strip():
                raise ValueError("empty English prompt or gloss sequence")
            gloss_count += len([g for g in row["gloss_sequence"].split("|") if g.strip()])
            phrase_id = row["phrase_id"].strip()
            if row["parallel"].strip().lower() == "yes":
                report["parallel_rows"] += 1
                phrase_signers[phrase_id].add(signer)
                signer_phrase_ids[signer]["parallel"].add(phrase_id)
                signer_row_counts[signer]["parallel"] += 1
            else:
                signer_phrase_ids[signer]["nonparallel"].add(phrase_id)
                signer_row_counts[signer]["nonparallel"] += 1
            if root is not None:
                ann_path = root / "annotations" / f"{clip}.json"
                if not ann_path.is_file():
                    raise ValueError("annotation JSON missing")
                annotation = json.loads(ann_path.read_text(encoding="utf-8"))
                for key in ("clip_id", "signer_id", "phrase_id", "text_en", "n_frames", "fps"):
                    expected = row[key].strip()
                    actual = str(annotation.get(key, "")).strip()
                    if key == "fps":
                        if not math.isclose(float(actual), fps, rel_tol=1e-6):
                            raise ValueError("annotation FPS mismatch")
                    elif key == "n_frames":
                        if int(actual) != frames:
                            raise ValueError("annotation frame-count mismatch")
                    elif actual != expected:
                        raise ValueError(f"annotation {key} mismatch")
                if bool(annotation.get("parallel")) != (row["parallel"].lower() == "yes"):
                    raise ValueError("annotation parallel mismatch")
                if str(annotation.get("hand_slots", "")).strip() != row["hand_slots"].strip():
                    raise ValueError("annotation hand_slots mismatch")
                segments = annotation.get("segments")
                if not isinstance(segments, list) or not segments:
                    raise ValueError("missing gloss segments")
                last_start = -1.0
                segment_glosses = []
                for segment in segments:
                    gloss = str(segment.get("gloss", "")).strip()
                    start, end = float(segment["start"]), float(segment["end"])
                    if not gloss or not math.isfinite(start + end) or start < 0 or end <= start or end > frames / fps + 1 / fps:
                        raise ValueError("invalid gloss or segment timing")
                    if start < last_start:
                        raise ValueError("gloss segments are not ordered")
                    last_start = start
                    segment_glosses.append(" ".join(gloss.upper().split()))
                expected_glosses = " ".join(" ".join(row["gloss_sequence"].upper().split()).split("|"))
                if " ".join(segment_glosses) != " ".join(expected_glosses.split()):
                    raise ValueError("metadata/annotation gloss sequence mismatch")
            if check_remote_headers:
                remote_candidates.append((index, clip, frames))
        except Exception as exc:
            report["errors"].append({"row": index, "clip_id": clip, "error": str(exc)})

    if check_remote_headers:
        def inspect_header(item):
            row_index, clip_id, frame_count = item
            header = fetch(f"{BASE}/keypoints/{clip_id}.npy", "bytes=0-65535", limit=65536)
            shape = npy_shape_from_header(header)
            if shape != (frame_count, 128, 3):
                raise ValueError(f"keypoint header shape {shape}, expected {(frame_count, 128, 3)}")
            return shape

        # Sample evenly across the corpus: one request per clip triggers upstream
        # rate limits and is unnecessary for validating the shared NPY schema.
        sample_size = min(24, len(remote_candidates))
        if sample_size:
            stride = max(1, len(remote_candidates) // sample_size)
            selected = remote_candidates[::stride][:sample_size]
        else:
            selected = []
        report["keypoint_header_sampled"] = len(selected)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(inspect_header, item): item for item in selected}
            for future in as_completed(futures):
                row_index, clip_id, _ = futures[future]
                try:
                    shape = future.result()
                    shape_counts[str(shape)] += 1
                    report["keypoint_header_checks"] += 1
                except Exception as exc:
                    report["errors"].append({
                        "row": row_index, "clip_id": clip_id,
                        "error": f"remote keypoint header: {exc}",
                    })

    report["unique_clip_ids"] = len(ids)
    report["unique_signers"] = len(report["signer_counts"])
    report["gloss_segment_count_from_metadata"] = gloss_count
    report["clips_per_signer"] = report["signer_counts"]
    report["oval_detection_summary"] = ({
        "min": min(oval_scores), "max": max(oval_scores),
        "mean": sum(oval_scores) / len(oval_scores),
    } if oval_scores else None)
    report["parallel_phrase_coverage"] = dict(sorted(Counter(len(v) for v in phrase_signers.values()).items()))
    # Exact assignments support a data-grounded signer-disjoint split.
    report["parallel_phrase_signers"] = {
        phrase: sorted(signers) for phrase, signers in sorted(phrase_signers.items())
    }
    report["signer_phrase_coverage"] = {
        signer: {
            "parallel_rows": signer_row_counts[signer]["parallel"],
            "nonparallel_rows": signer_row_counts[signer]["nonparallel"],
            "parallel_phrase_ids": sorted(signer_phrase_ids[signer]["parallel"]),
            "nonparallel_phrase_ids": sorted(signer_phrase_ids[signer]["nonparallel"]),
        }
        for signer in sorted(signer_row_counts)
    }
    if len(rows) != 2400:
        report["warnings"].append(f"expected 2400 clips for v0.4, found {len(rows)}")
    report["keypoint_shape_counts"] = dict(shape_counts)
    if len(ids) != len(rows):
        report["warnings"].append("clip IDs are not all unique")
    if check_remote_headers and report["keypoint_header_checks"] != len(rows):
        report["warnings"].append(
            "remote NPY header schema was sampled; not every clip header was fetched"
        )
    report["error_summary"] = dict(sorted(Counter(
        (item.get("error", "unknown") if isinstance(item, dict) else str(item))
        for item in report["errors"]
    ).items()))
    report["sample_errors"] = report["errors"][:20]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, help="local metadata.csv; otherwise fetch public metadata")
    parser.add_argument("--dataset-root", type=Path, help="optional local annotations/ directory for full JSON checks")
    parser.add_argument("--remote-keypoint-headers", action="store_true",
                        help="HTTP Range-read NPY headers only; no keypoint arrays are downloaded")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.metadata.read_bytes() if args.metadata else fetch(f"{BASE}/metadata.csv")
    report = audit(args.dataset_root, data, args.remote_keypoint_headers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"errors"}}, indent=2))
    print(f"errors: {len(report['errors'])}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
