import unittest

from audit_asl_corpus import audit_rows


def row(clip, signer, split):
    return {
        "source_dataset": "sample-corpus",
        "source_clip_id": clip,
        "signer_id": signer,
        "split": split,
        "start_ms": "0",
        "end_ms": "2500",
        "transcript": "HELLO",
        "transcript_kind": "gloss",
        "modality": "rgb",
        "license_id": "permission-record-1",
        "permission_status": "approved",
        "attribution": "Dataset authors",
        "source_url": "https://example.org/corpus",
        "consent": "yes",
    }


class AuditAslCorpusTests(unittest.TestCase):
    def test_reports_split_signer_and_duration_coverage(self):
        report = audit_rows([
            row("a", "signer-a", "train"),
            row("b", "signer-b", "validation"),
            row("c", "signer-c", "test"),
        ])
        self.assertTrue(report["ok"])
        self.assertEqual(report["clips"], 3)
        self.assertEqual(report["unique_signers"], 3)
        self.assertEqual(report["clips_by_split"], {
            "train": 1, "validation": 1, "test": 1
        })
        self.assertEqual(report["signers_by_split"], {
            "train": 1, "validation": 1, "test": 1
        })
        self.assertEqual(report["clip_duration_ms"]["median"], 2500)

    def test_invalid_manifest_returns_validation_errors(self):
        invalid = row("a", "signer-a", "train")
        invalid["permission_status"] = "pending"
        report = audit_rows([invalid])
        self.assertFalse(report["ok"])
        self.assertTrue(any("permission_status must be approved" in e for e in report["errors"]))


if __name__ == "__main__":
    unittest.main()
