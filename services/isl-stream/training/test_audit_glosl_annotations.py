import csv
import tempfile
import unittest
from pathlib import Path

from training.audit_glosl_annotations import audit_annotations


class AuditGloslAnnotationsTests(unittest.TestCase):
    def test_reports_gloss_types_and_signer_video_split_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.csv"
            fields = ["id", "split", "text", "gloss", "gloss_type", "signer", "video_ref", "start", "end", "fps", "n_frames"]
            rows = [
                ["a", "train", "hello", "HELLO", "manual", "signer-1", "video-a", "0.0", "1.2", "25", "30"],
                ["b", "test", "hello", "HELLO", "manual", "signer-1", "video-b", "1.2", "2.0", "25", "20"],
                ["c", "dev", "thanks", "", "", "signer-2", "video-c", "bad", "3.0", "0", "-1"],
                ["d", "test", "please", "PLEASE", "manual", "signer-3", "video-a", "2.0", "1.0", "25", "10"],
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            report = audit_annotations(path)
            self.assertEqual(report["rows"], 4)
            self.assertEqual(report["gloss_type_counts"]["manual"], 3)
            self.assertEqual(report["empty_gloss_rows"], 1)
            self.assertEqual(report["signers_in_multiple_splits"], 1)
            self.assertEqual(report["videos_in_multiple_splits"], 1)
            self.assertEqual(report["invalid_timestamp_rows"], 2)\n            self.assertEqual(report["invalid_fps_rows"], 1)\n            self.assertEqual(report["invalid_frame_count_rows"], 1)\n            self.assertEqual(report["missing_video_ref_rows"], 0)\n            self.assertFalse(report["training_approved"])

    def test_reports_duplicate_ids_and_invalid_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.csv"
            fields = ["id", "split", "text", "gloss", "gloss_type", "signer"]
            rows = [
                ["same", "train", "hello", "HELLO", "manual", "signer-1"],
                ["same", "holdout", "hello", "HELLO", "manual", "signer-2"],
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            report = audit_annotations(path)
            self.assertEqual(report["duplicate_ids"], 1)
            self.assertEqual(report["invalid_split_rows"], 1)
            self.assertEqual(report["split_counts"]["holdout"], 1)

    def test_rejects_missing_required_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.csv"
            path.write_text("id,split,text\na,train,hello\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                audit_annotations(path)


if __name__ == "__main__":
    unittest.main()
