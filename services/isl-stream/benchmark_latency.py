"""CLI for summarising SIGNBRIDGE phrase-level benchmark JSONL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from latency_metrics import summarize_latency


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report p50/p95 sign-stream latency and dropped-frame metrics."
    )
    parser.add_argument("input", type=Path, help="JSONL file containing one phrase record per line")
    args = parser.parse_args()

    records = []
    try:
        with args.input.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"line {line_number}: invalid JSON ({exc.msg})") from None
        report = summarize_latency(records)
    except (OSError, ValueError) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
