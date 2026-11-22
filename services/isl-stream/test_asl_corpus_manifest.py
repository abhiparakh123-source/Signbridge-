import unittest
from training.asl_corpus_manifest import validate_rows


def row(**overrides):
    values = {
        "source_dataset": "How2Sign",
        "source_clip_id": "clip-001",
        "signer_id": "signer-private-01",
        "split": "train",
        "start_ms": "1000",
        "end_ms": "5000",
        "transcript": "How are you?",
        "transcript_kind": "translation",
        "modality": "rgb+pose",
        "license_id": "CC-BY-NC-4.0",
        "permission_status": "approved",
        "attribution": "How2Sign project",
        "source_url": "https://how2sign.github.io/",
    }
    values.update(overrides)
    return values


class AslCorpusManifestTests(unittest.TestCase):
    def test_valid_continuous_translation_row_passes(self):
        self.assertEqual(validate_rows([row()]), [])

    def test_transcript_kind_must_be_explicit(self):
        self.assertTrue(any("transcript_kind" in e for e in validate_rows([row(transcript_kind="english")])))

    def test_unapproved_source_is_rejected(self):
        self.assertTrue(any("permission_status" in e for e in validate_rows([row(permission_status="pending")])))

    def test_invalid_time_range_is_rejected(self):
        self.assertTrue(any("start_ms/end_ms" in e for e in validate_rows([row(start_ms="5000", end_ms="1000")])))

    def test_duplicate_source_clip_is_rejected(self):
        self.assertTrue(any("duplicate source clip" in e for e in validate_rows([row(), row()])))

    def test_signer_cannot_cross_splits(self):
        self.assertTrue(any("multiple splits" in e for e in validate_rows([
            row(source_clip_id="a", split="train"),
            row(source_clip_id="b", split="test"),
        ])))

    def test_empty_manifest_is_rejected(self):
        self.assertTrue(validate_rows([]))


if __name__ == "__main__":
    import unittest
    unittest.main()
