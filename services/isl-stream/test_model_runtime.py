"""Tests for fail-closed checkpoint readiness in the API runtime."""
import tempfile
import unittest
from pathlib import Path

import torch

from model_runtime import ModelRuntime
from training.model import LandmarkCTCModel


class ModelRuntimeTests(unittest.TestCase):
    def make_checkpoint(self, path, language="ASL"):
        model = LandmarkCTCModel(
            feature_dim=3, vocabulary_size=2, width=8,
            heads=2, layers=1, dropout=0.0, causal=True,
        )
        torch.save({
            "state_dict": model.state_dict(),
            "feature_dim": 3,
            "vocabulary": {"<blank>": 0, "HELLO": 1},
            "sign_language": language,
            "model_type": "landmark_ctc_transformer_baseline",
            "architecture": {
                "width": 8, "heads": 2, "layers": 1,
                "dropout": 0.0, "causal": True,
            },
        }, path)

    def test_unconfigured_runtime_fails_closed(self):
        runtime = ModelRuntime(checkpoint_path="", expected_language="ASL")
        status = runtime.status(attempt_load=True)
        self.assertFalse(status["model_loaded"])
        self.assertFalse(status["recognition_enabled"])
        self.assertFalse(status["translation_enabled"])
        self.assertEqual(status["status"], "model_not_configured")

    def test_missing_checkpoint_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pt"
            runtime = ModelRuntime(str(missing), "ASL")
            status = runtime.status(attempt_load=True)
            self.assertFalse(status["model_loaded"])
            self.assertEqual(status["status"], "checkpoint_file_missing_or_invalid")

    def test_language_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "asl.pt"
            self.make_checkpoint(checkpoint, "ASL")
            runtime = ModelRuntime(str(checkpoint), "ISL")
            status = runtime.status(attempt_load=True)
            self.assertFalse(status["model_loaded"])
            self.assertEqual(status["status"], "checkpoint_language_mismatch")
            self.assertFalse(status["recognition_enabled"])

    def test_loaded_checkpoint_does_not_enable_live_recognition(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "asl.pt"
            self.make_checkpoint(checkpoint, "ASL")
            runtime = ModelRuntime(str(checkpoint), "ASL")
            status = runtime.status(attempt_load=True)
            self.assertTrue(status["model_loaded"])
            self.assertEqual(status["checkpoint_language"], "ASL")
            self.assertEqual(status["status"], "checkpoint_loaded_streaming_not_integrated")
            self.assertFalse(status["recognition_enabled"])
            self.assertFalse(status["translation_enabled"])

    def test_rejects_unknown_expected_language(self):
        with self.assertRaisesRegex(ValueError, "SIGNBRIDGE_MODEL_LANGUAGE"):
            ModelRuntime(checkpoint_path="", expected_language="Hindi")


if __name__ == "__main__":
    unittest.main()
