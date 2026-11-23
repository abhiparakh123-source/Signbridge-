from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare_epee_keypoints import (
    clean_gloss, convert_dataset, read_annotations, split_signers,
    validate_permission_record,
)


class EpeeAdapterTests(unittest.TestCase):
    def _approved_permission_record(self, path: Path):
        path.write_text(json.dumps({
            "status": "approved",
            "dataset_id": "CLERC-DATA/epee",
            "license_id": "CC-BY-NC-SA-4.0",
            "approved_by": "Test rights reviewer",
            "reviewed_at": "2026-11-23",
            "written_permission_reference": "https://example.org/permission-letter",
            "approved_uses": [
                "download_dataset", "create_derived_features", "train_model",
                "retain_model_checkpoint", "deploy_private_non_revenue_family_app",
            ],
        }), encoding="utf-8")

    def test_permission_record_must_be_approved_and_cover_all_uses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "permission.json"
            path.write_text(json.dumps({"status": "pending"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be approved"):
                validate_permission_record(path)
            self._approved_permission_record(path)
            self.assertEqual(validate_permission_record(path)["status"], "approved")

    def test_user_attestation_is_recorded_without_fabricated_identity_or_url(self):
        record = validate_permission_record(None, user_attested=True)
        self.assertEqual(record["status"], "user_attested")
        self.assertEqual(record["authorization_basis"], "user_confirmation_in_conversation")
        self.assertNotIn("approved_by", record)
        self.assertNotIn("written_permission_reference", record)

    def test_permission_record_rejects_missing_checkpoint_rights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "permission.json"
            self._approved_permission_record(path)
            record = json.loads(path.read_text(encoding="utf-8"))
            record["approved_uses"].remove("retain_model_checkpoint")
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "retain_model_checkpoint"):
                validate_permission_record(path)

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
            permission = Path(directory) / "permission.json"
            self._approved_permission_record(permission)
            result = convert_dataset(root, output, ["A"], ["B"], ["C"], permission)
            self.assertEqual(result["clips"], 3)
            self.assertEqual(result["feature_dim"], 384)
            feature = np.load(output / "features" / "c_train.npy", allow_pickle=False)
            self.assertEqual(feature.shape, (8, 384))
            self.assertTrue((output / "features_manifest.csv").is_file())


if __name__ == "__main__":
    unittest.main()
