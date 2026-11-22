"""CLI integration tests for the JSONL latency benchmark."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("benchmark_latency.py")


class BenchmarkLatencyCliTests(unittest.TestCase):
    def test_valid_jsonl_prints_machine_readable_report(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.jsonl"
            source.write_text(
                json.dumps({
                    "captured_at_ms": 10,
                    "first_provisional_at_ms": 25,
                    "stable_at_ms": 50,
                    "translated_at_ms": 70,
                    "captured_frames": 20,
                    "dropped_frames": 2,
                }) + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(source)],
                capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["phrases"], 1)
        self.assertEqual(report["first_provisional_latency_ms"]["p50"], 15)
        self.assertEqual(report["dropped_frame_percent"], 10.0)

    def test_malformed_jsonl_returns_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "broken.jsonl"
            source.write_text('{"captured_at_ms": 1}\nnot-json\n', encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(source)],
                capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("line 2: invalid JSON", result.stderr)


if __name__ == "__main__":
    unittest.main()
