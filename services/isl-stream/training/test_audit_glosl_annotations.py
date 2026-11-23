import csv
import tempfile
import unittest
from pathlib import Path

from training.audit_glosl_annotations import audit_annotations


class AuditGloslAnnotationsTests(unittest.TestCase):
    def test_reports_gloss_types_and_split_leakage_without_approving_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.csv"
            fields = ["id", "split", "text", "gloss", "gloss_type", "signer", "video_ref"]
            rows = [
                ["a", "train", "hello", "HELLO", "manual", "signer-1", "video-a"],
                ["b", "test", "hello", "HELLO", "manual", "signer-1", "video-b"],
                ["c", "dev", "thanks", "", "", "signer-2", "video-c"],
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            report = audit_annotations(path)
            self.assertEqual(report["rows"], 3)
            self.assertEqual(report["gloss_type_counts"]["manual"], 2)
            self.assertEqual(report["empty_gloss_rows"], 1)
            self.assertEqual(report["signers_in_multiple_splits"], 1)
            self.assertFalse(report["training_approved"])

    def test_rejects_missing_required_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.csv"
            path.write_text("id,split,text\na,train,hello\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                audit_annotations(path)


if __name__ == "__main__":
    unittest.main()
