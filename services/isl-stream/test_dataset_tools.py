import csv
import tempfile
import unittest
from pathlib import Path
from dataset_tools import validate


class DatasetManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "clips").mkdir()
        (self.root / "clips" / "a.webm").write_bytes(b"test")
        self.manifest = self.root / "manifest.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def write_rows(self, rows):
        fields = ["clip_id", "signer_id", "sign_language", "label", "video_path", "consent", "split"]
        with self.manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def row(self, **overrides):
        data = dict(clip_id="a", signer_id="signer-a", sign_language="ISL", label="hello",
                    video_path="clips/a.webm", consent="yes", split="train")
        data.update(overrides)
        return data

    def test_valid_manifest_passes(self):
        self.write_rows([self.row()])
        self.assertEqual(validate(self.manifest, self.root), [])

    def test_signer_cannot_leak_across_splits(self):
        self.write_rows([self.row(clip_id="a", split="train"),
                         self.row(clip_id="b", split="test")])
        self.assertTrue(any("multiple splits" in e for e in validate(self.manifest, self.root)))

    def test_missing_consent_fails(self):
        self.write_rows([self.row(consent="no")])
        self.assertTrue(any("consent" in e for e in validate(self.manifest, self.root)))

    def test_unknown_sign_language_fails(self):
        self.write_rows([self.row(sign_language="XYZ")])
        self.assertTrue(any("sign_language" in e for e in validate(self.manifest, self.root)))

    def test_path_traversal_fails(self):
        self.write_rows([self.row(video_path="../outside.webm")])
        self.assertTrue(any("stay inside" in e for e in validate(self.manifest, self.root)))

    def test_reused_video_path_is_rejected(self):
        self.write_rows([
            self.row(clip_id="a", split="train"),
            self.row(clip_id="b", split="test"),
        ])
        errors = validate(self.manifest, self.root)
        self.assertTrue(any("video file is reused" in e for e in errors))

    def test_duplicate_video_content_is_rejected_across_paths(self):
        (self.root / "clips" / "b.webm").write_bytes(b"test")
        self.write_rows([
            self.row(clip_id="a", video_path="clips/a.webm", split="train"),
            self.row(clip_id="b", video_path="clips/b.webm", split="test"),
        ])
        errors = validate(self.manifest, self.root)
        self.assertTrue(any("video content duplicates" in e for e in errors))

    def test_empty_video_is_rejected(self):
        (self.root / "clips" / "a.webm").write_bytes(b"")
        self.write_rows([self.row()])
        self.assertTrue(any("video file is empty" in e for e in validate(self.manifest, self.root)))


if __name__ == "__main__":
    unittest.main()
