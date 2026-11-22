"""Dependency-free tests for experimental checkpoint metadata validation."""
import unittest

from training.checkpoint_validation import validate_checkpoint_metadata


class CheckpointValidationTests(unittest.TestCase):
    def checkpoint(self, **overrides):
        value = {
            "model_type": "landmark_ctc_transformer_baseline",
            "vocabulary": {"<blank>": 0, "HELLO": 1},
            "feature_dim": 1699,
            "sign_language": "ISL",
            "architecture": {"width": 256, "heads": 4, "layers": 4, "dropout": 0.1},
        }
        value.update(overrides)
        return value

    def test_valid_metadata_returns_normalized_architecture(self):
        vocabulary, feature_dim, language, architecture = validate_checkpoint_metadata(self.checkpoint())
        self.assertEqual(vocabulary["HELLO"], 1)
        self.assertEqual(feature_dim, 1699)
        self.assertEqual(language, "ISL")
        self.assertEqual(architecture["dropout"], 0.1)
        self.assertFalse(architecture["causal"])

    def test_rejects_unknown_checkpoint_type(self):
        with self.assertRaisesRegex(ValueError, "checkpoint type"):
            validate_checkpoint_metadata(self.checkpoint(model_type="unknown"))

    def test_rejects_missing_blank_or_noncontiguous_vocabulary(self):
        for vocabulary in ({"HELLO": 0}, {"<blank>": 0, "HELLO": 2}, {"<blank>": 1}):
            with self.subTest(vocabulary=vocabulary), self.assertRaises(ValueError):
                validate_checkpoint_metadata(self.checkpoint(vocabulary=vocabulary))

    def test_rejects_invalid_vocabulary_keys_and_ids(self):
        for vocabulary in (
            {"<blank>": 0, "": 1},
            {"<blank>": 0, "HELLO": True},
            {"<blank>": 0, "HELLO": "1"},
        ):
            with self.subTest(vocabulary=vocabulary), self.assertRaises(ValueError):
                validate_checkpoint_metadata(self.checkpoint(vocabulary=vocabulary))

    def test_rejects_invalid_feature_dimension_and_language(self):
        for feature_dim in (0, -1, True, 1.5):
            with self.subTest(feature_dim=feature_dim), self.assertRaises(ValueError):
                validate_checkpoint_metadata(self.checkpoint(feature_dim=feature_dim))
        with self.assertRaisesRegex(ValueError, "sign_language"):
            validate_checkpoint_metadata(self.checkpoint(sign_language="Hindi"))

    def test_rejects_unhashable_language_metadata_cleanly(self):
        with self.assertRaisesRegex(ValueError, "sign_language"):
            validate_checkpoint_metadata(self.checkpoint(sign_language=[]))

    def test_accepts_causal_architecture_metadata(self):
        architecture = {
            "width": 256, "heads": 4, "layers": 4,
            "dropout": 0.1, "causal": True,
        }
        result = validate_checkpoint_metadata(self.checkpoint(architecture=architecture))
        self.assertTrue(result[3]["causal"])

    def test_rejects_non_boolean_causal_metadata(self):
        for value in (1, 0, "true", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "causal"):
                validate_checkpoint_metadata(self.checkpoint(
                    architecture={"causal": value}
                ))

    def test_rejects_invalid_architecture(self):
        for architecture in (
            [],
            {"width": 0},
            {"heads": True},
            {"width": 255, "heads": 4},
            {"dropout": 1.0},
            {"dropout": float("nan")},
        ):
            with self.subTest(architecture=architecture), self.assertRaises(ValueError):
                validate_checkpoint_metadata(self.checkpoint(architecture=architecture))


if __name__ == "__main__":
    unittest.main()
