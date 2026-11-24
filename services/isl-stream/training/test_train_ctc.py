"""Synthetic-fixture tests for the CTC training data and epoch pipeline.

These tests validate software plumbing only. Synthetic features are not training
data and must never be used to claim sign-language recognition quality.
"""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from training.train_ctc import (
    LandmarkDataset,
    LandmarkNormalizer,
    collate,
    evaluate,
    load_manifest,
    run_epoch,
)


class TinyTrainableModel(nn.Module):
    """Small differentiable frame classifier used only to exercise trainer loops."""
    def __init__(self, feature_dim, classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, classes)

    def forward(self, features, lengths):
        logits = self.classifier(features).log_softmax(dim=-1)
        return logits.transpose(0, 1), lengths


class TrainerPipelineTests(unittest.TestCase):
    FIELDS = [
        "clip_id", "signer_id", "sign_language", "feature_path",
        "gloss", "consent", "split",
    ]

    def make_dataset(self, root):
        records = [
            ("train-1", "signer-train", "train"),
            ("validation-1", "signer-validation", "validation"),
            ("test-1", "signer-test", "test"),
        ]
        rows = []
        for clip_id, signer, split in records:
            feature_path = f"{clip_id}.npy"
            np.save(root / feature_path, np.ones((5, 3), dtype=np.float32))
            rows.append({
                "clip_id": clip_id,
                "signer_id": signer,
                "sign_language": "ASL",
                "feature_path": feature_path,
                "gloss": "HELLO",
                "consent": "yes",
                "split": split,
            })
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def test_manifest_dataset_collation_and_train_eval_loops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_dataset(root)
            rows = load_manifest(manifest, root, "ASL")
            vocabulary = {"<blank>": 0, "<unk>": 1, "HELLO": 2}
            train_rows = [row for row in rows if row["_split"] == "train"]
            dataset = LandmarkDataset(train_rows, vocabulary)
            features, targets = dataset[0]
            self.assertEqual(tuple(features.shape), (5, 3))
            self.assertEqual(targets.tolist(), [2])

            loader = DataLoader(dataset, batch_size=1, collate_fn=collate)
            model = TinyTrainableModel(3, 3)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
            train_loss = run_epoch(model, loader, loss_fn, optimizer, torch.device("cpu"))
            self.assertTrue(np.isfinite(train_loss))

            reverse = {0: "<blank>", 1: "<unk>", 2: "HELLO"}
            metrics = evaluate(model, loader, loss_fn, torch.device("cpu"), reverse)
            self.assertEqual(metrics["clips"], 1)
            self.assertTrue(np.isfinite(metrics["ctc_loss"]))
            self.assertGreaterEqual(metrics["gloss_wer"], 0.0)
            self.assertGreater(metrics["predicted_frame_count"], 0)
            self.assertGreaterEqual(metrics["blank_frame_fraction"], 0.0)
            self.assertLessEqual(metrics["blank_frame_fraction"], 1.0)
            self.assertGreaterEqual(metrics["decoded_gloss_token_count"], 0)
            self.assertIn("top_predicted_frame_classes", metrics)
            self.assertIn("top_decoded_gloss_tokens", metrics)
            self.assertLessEqual(len(metrics["top_predicted_frame_classes"]), 12)
            self.assertLessEqual(len(metrics["top_decoded_gloss_tokens"]), 12)

    def test_normalizer_uses_observed_training_points_and_preserves_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "train.npy"
            # Two observed landmarks; one absent landmark represented by zeros.
            values = np.array([
                [1, 2, 3] * 127 + [0, 0, 0],
                [3, 4, 5] * 127 + [0, 0, 0],
            ], dtype=np.float32)
            np.save(path, values)
            rows = [{"_path": path}]
            normalizer = LandmarkNormalizer.fit(rows)
            transformed = normalizer.transform(values)
            self.assertTrue(np.isfinite(transformed).all())
            self.assertTrue(np.array_equal(transformed[:, -3:], np.zeros((2, 3), dtype=np.float32)))
            observed = transformed[:, :3]
            self.assertAlmostEqual(float(observed[:, 0].mean()), 0.0, places=5)
            self.assertAlmostEqual(float(observed[:, 1].mean()), 0.0, places=5)
            self.assertAlmostEqual(float(observed[:, 2].mean()), 0.0, places=5)

    def test_normalizer_rejects_non_384_dimensional_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "bad.npy"
            np.save(path, np.ones((4, 12), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "384-D"):
                LandmarkNormalizer.fit([{"_path": path}])

    def test_maps_heldout_gloss_to_explicit_unknown_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_dataset(root)
            rows = load_manifest(manifest, root, "ASL")
            validation_rows = [row for row in rows if row["_split"] == "validation"]
            # HELLO is deliberately absent from the training vocabulary.
            dataset = LandmarkDataset(validation_rows, {"<blank>": 0, "<unk>": 1, "TRAIN_ONLY": 2})
            _, targets = dataset[0]
            self.assertEqual(targets.tolist(), [1])
            loader = DataLoader(dataset, batch_size=1, collate_fn=collate)
            model = TinyTrainableModel(3, 3)
            metrics = evaluate(
                model,
                loader,
                nn.CTCLoss(blank=0, zero_infinity=True),
                torch.device("cpu"),
                {0: "<blank>", 1: "<unk>", 2: "TRAIN_ONLY"},
            )
            self.assertEqual(metrics["oov_reference_tokens"], 1)
            self.assertEqual(metrics["oov_reference_clips"], 1)
            self.assertEqual(metrics["in_vocabulary_only_clips"], 0)

    def test_rejects_signer_leakage_between_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_dataset(root)
            with manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["signer_id"] = rows[0]["signer_id"]
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "appears in multiple splits"):
                load_manifest(manifest, root, "ASL")

    def test_rejects_feature_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_dataset(root)
            with manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["feature_path"] = "../outside.npy"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "escapes dataset root"):
                load_manifest(manifest, root, "ASL")

    def test_rejects_missing_explicit_consent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_dataset(root)
            with manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["consent"] = "no"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "no explicit consent"):
                load_manifest(manifest, root, "ASL")


if __name__ == "__main__":
    unittest.main()
