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
            vocabulary = {"<blank>": 0, "HELLO": 1}
            train_rows = [row for row in rows if row["_split"] == "train"]
            dataset = LandmarkDataset(train_rows, vocabulary)
            features, targets = dataset[0]
            self.assertEqual(tuple(features.shape), (5, 3))
            self.assertEqual(targets.tolist(), [1])

            loader = DataLoader(dataset, batch_size=1, collate_fn=collate)
            model = TinyTrainableModel(3, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
            train_loss = run_epoch(model, loader, loss_fn, optimizer, torch.device("cpu"))
            self.assertTrue(np.isfinite(train_loss))

            reverse = {0: "<blank>", 1: "HELLO"}
            metrics = evaluate(model, loader, loss_fn, torch.device("cpu"), reverse)
            self.assertEqual(metrics["clips"], 1)
            self.assertTrue(np.isfinite(metrics["ctc_loss"]))
            self.assertGreaterEqual(metrics["gloss_wer"], 0.0)

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
