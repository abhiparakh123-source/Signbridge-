from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare_epee_keypoints import (
    clean_gloss, convert_dataset, read_annotations, split_signers,
)


class EpeeAdapterTests(unittest.TestCase):
    def test_clean_gloss_normalizes_whitespace_and_case(self):
        self.assertEqual(clean_gloss("  what   UP  "), "WHAT UP")

    def test_split_signers_rejects_overlap(self):
        with self.assertRaisesRegex(ValueError, "appears in both"):
            split_signers(["A"], ["A"], ["C"])

    def test_annotation_identity_and_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.json"
            row = {"clip_id": "c1", "signer_id": "A", "phrase_id": "p1",
                   "text_en": "Hello", "parallel": "yes", "gloss_sequence": "HELLO"}
            path.write_text(json.dumps({
                "clip_id": "c1", "signer_id": "A", "sign_language": "ASL",
                "phrase_id": "p1", "text_en": "Hello", "parallel": True,
                "fps": 30.0, "n_frames": 30,
                "segments": [{"gloss": "HELLO", "start": 0.1, "end": 0.5}],
            }), encoding="utf-8")
            self.assertEqual(read_annotations(path, row, 30, 30.0), "HELLO")
            row["clip_id"] = "c2"
            with self.assertRaisesRegex(ValueError, "identity"):
                read_annotations(path, row, 30, 30.0)

    def test_end_to_end_creates_384_dim_features_and_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory) / "dataset", Path(directory) / "out"
            (root / "annotations").mkdir(parents=True)
            (root / "keypoints").mkdir()
            rows = []
            for clip, signer, phrase in [
                ("c_train", "A", "p1"), ("c_val", "B", "p1"), ("c_test", "C", "p1")
            ]:
                rows.append({
                    "clip_id": clip, "signer_id": signer, "phrase_id": phrase,
                    "fps": "30", "n_frames": "8", "hand_slots": "verified",
                    "parallel": "yes", "text_en": "Hello", "gloss_sequence": "HELLO",
                })
                (root / "annotations" / f"{clip}.json").write_text(json.dumps({
                    "clip_id": clip, "signer_id": signer, "sign_language": "ASL",
                    "phrase_id": phrase, "text_en": "Hello", "parallel": True,
                    "fps": 30.0, "n_frames": 8,
                    "segments": [{"gloss": "HELLO", "start": 0.0, "end": 0.2}],
                }), encoding="utf-8")
                np.save(root / "keypoints" / f"{clip}.npy",
                        np.ones((8, 128, 3), dtype=np.float32))
            with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
                import csv
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = convert_dataset(root, output, ["A"], ["B"], ["C"])
            self.assertEqual(result["clips"], 3)
            self.assertEqual(result["feature_dim"], 384)
            feature = np.load(output / "features" / "c_train.npy", allow_pickle=False)
            self.assertEqual(feature.shape, (8, 384))
            self.assertTrue((output / "features_manifest.csv").is_file())


if __name__ == "__main__":
    unittest.main()
