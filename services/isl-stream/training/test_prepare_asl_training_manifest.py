import tempfile
import unittest
from pathlib import Path

from training.prepare_asl_training_manifest import convert_rows


def row(dataset="corpus", clip="clip-1", signer="s1", split="train", kind="gloss"):
    return {
        "source_dataset": dataset, "source_clip_id": clip, "signer_id": signer,
        "split": split, "start_ms": "0", "end_ms": "1200",
        "transcript": "HELLO WORLD", "transcript_kind": kind,
        "modality": "rgb", "license_id": "license-x",
        "permission_status": "approved", "attribution": "Dataset authors",
        "source_url": "https://example.org/dataset", "media_path": "clips/a.mp4",
        "consent": "yes",
    }


class PrepareAslManifestTests(unittest.TestCase):
    def test_converts_only_gloss_rows_and_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clips").mkdir()
            (root / "clips/a.mp4").write_bytes(b"video")
            rows = [
                row(split="train"),
                row(clip="clip-2", signer="s2", split="validation"),
                row(clip="clip-3", signer="s3", split="test"),
            ]
            result = convert_rows(rows, root)
            self.assertEqual(result[0]["sign_language"], "ASL")
            self.assertEqual(result[0]["label"], "HELLO WORLD")
            self.assertEqual(result[0]["source_clip_id"], "clip-1")
            self.assertEqual(result[0]["clip_id"], "corpus__clip-1")

    def test_refuses_english_translation_as_gloss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clips").mkdir()
            (root / "clips/a.mp4").write_bytes(b"video")
            with self.assertRaisesRegex(ValueError, "must not be relabeled"):
                convert_rows([row(kind="translation")], root)

    def test_refuses_unapproved_permission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clips").mkdir()
            (root / "clips/a.mp4").write_bytes(b"video")
            item = row()
            item["permission_status"] = "unknown"
            with self.assertRaisesRegex(ValueError, "not approved"):
                convert_rows([item], root)

    def test_refuses_signer_leakage_across_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clips").mkdir()
            (root / "clips/a.mp4").write_bytes(b"video")
            rows = [row(split="train"), row(clip="clip-2", split="test")]
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                convert_rows(rows, root)

    def test_refuses_media_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = row()
            item["media_path"] = "../outside.mp4"
            with self.assertRaisesRegex(ValueError, "outside"):
                convert_rows([item], root)


if __name__ == "__main__":
    unittest.main()
