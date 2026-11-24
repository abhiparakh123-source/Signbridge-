"""Real checkpoint round-trip tests using a tiny randomly initialized model.

This verifies serialization/loading and shape compatibility only. It does not
measure or imply sign-language recognition accuracy.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from training.inference import load_model, normalize_landmark_features, predict
from training.model import LandmarkCTCModel


class InferenceCheckpointLoadingTests(unittest.TestCase):
    def test_loads_weights_only_checkpoint_and_runs_gloss_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny-asl.pt"
            vocabulary = {"<blank>": 0, "HELLO": 1}
            model = LandmarkCTCModel(
                feature_dim=3, vocabulary_size=2, width=8,
                heads=2, layers=1, dropout=0.0, causal=True,
            )
            checkpoint = {
                "state_dict": model.state_dict(),
                "feature_dim": 3,
                "vocabulary": vocabulary,
                "sign_language": "ASL",
                "model_type": "landmark_ctc_transformer_baseline",
                "architecture": {
                    "width": 8, "heads": 2, "layers": 1,
                    "dropout": 0.0, "causal": True,
                },
            }
            torch.save(checkpoint, path)
            loaded, loaded_vocabulary, metadata = load_model(path, torch.device("cpu"))
            self.assertEqual(loaded_vocabulary, vocabulary)
            self.assertEqual(metadata["sign_language"], "ASL")
            self.assertTrue(loaded.causal)
            output = predict(
                np.zeros((6, 3), dtype=np.float32),
                loaded, torch.device("cpu"), loaded_vocabulary,
            )
            self.assertIsInstance(output, list)
            self.assertTrue(all(token in {"HELLO"} for token in output))


    def test_landmark_normalization_uses_checkpoint_stats_and_preserves_missing(self):
        raw = np.zeros((2, 384), dtype=np.float32)
        raw[0, :3] = [3.0, 5.0, 7.0]
        raw[1, :3] = [5.0, 7.0, 9.0]
        stats = {
            "method": "coordinate_zscore_observed_landmarks",
            "mean": [4.0, 6.0, 8.0],
            "scale": [2.0, 2.0, 2.0],
            "missing_landmark": "preserve_zero_triple",
        }
        normalized = normalize_landmark_features(raw, stats)
        np.testing.assert_allclose(normalized[0, :3], [-0.5, -0.5, -0.5])
        np.testing.assert_array_equal(normalized[:, 3:], 0.0)

    def test_rejects_invalid_checkpoint_normalization(self):
        with self.assertRaises(ValueError):
            normalize_landmark_features(
                np.zeros((2, 384), dtype=np.float32),
                {"method": "unknown", "mean": [0, 0, 0], "scale": [1, 1, 1],
                 "missing_landmark": "preserve_zero_triple"},
            )

    def test_rejects_checkpoint_with_incompatible_state_dict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.pt"
            checkpoint = {
                "state_dict": {"unexpected.weight": torch.zeros(1)},
                "feature_dim": 3,
                "vocabulary": {"<blank>": 0, "HELLO": 1},
                "sign_language": "ASL",
                "model_type": "landmark_ctc_transformer_baseline",
                "architecture": {"width": 8, "heads": 2, "layers": 1},
            }
            torch.save(checkpoint, path)
            with self.assertRaises(RuntimeError):
                load_model(path, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
