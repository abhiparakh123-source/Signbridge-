"""Checkpoint metadata validation tests (no torch dependency required)."""
import unittest

from training.checkpoint_validation import validate_checkpoint_metadata


def valid_checkpoint():
    return {
        "model_type": "landmark_ctc_transformer_baseline",
        "vocabulary": {"<blank>": 0, "HELLO": 1, "THANK-YOU": 2},
        "feature_dim": 1699,
        "sign_language": "ASL",
        "architecture": {
            "width": 256, "heads": 4, "layers": 4,
            "dropout": 0.1, "causal": True,
        },
    }


class CheckpointMetadataTests(unittest.TestCase):
    def test_accepts_valid_asl_checkpoint_and_returns_normalized_metadata(self):
        vocabulary, feature_dim, language, architecture = validate_checkpoint_metadata(valid_checkpoint())
        self.assertEqual(vocabulary["<blank>"], 0)
        self.assertEqual(feature_dim, 1699)
        self.assertEqual(language, "ASL")
        self.assertTrue(architecture["causal"])

    def test_rejects_wrong_model_type(self):
        checkpoint = valid_checkpoint()
        checkpoint["model_type"] = "sentence_translator"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_checkpoint_metadata(checkpoint)

    def test_rejects_noncontiguous_or_duplicate_vocabulary_ids(self):
        for vocabulary in (
            {"<blank>": 0, "HELLO": 2},
            {"<blank>": 0, "HELLO": 1, "WORLD": 1},
        ):
            checkpoint = valid_checkpoint()
            checkpoint["vocabulary"] = vocabulary
            with self.assertRaisesRegex(ValueError, "unique, contiguous"):
                validate_checkpoint_metadata(checkpoint)

    def test_rejects_boolean_dimensions(self):
        checkpoint = valid_checkpoint()
        checkpoint["feature_dim"] = True
        with self.assertRaisesRegex(ValueError, "positive integer"):
            validate_checkpoint_metadata(checkpoint)

    def test_rejects_unknown_language(self):
        checkpoint = valid_checkpoint()
        checkpoint["sign_language"] = "ENGLISH"
        with self.assertRaisesRegex(ValueError, "sign_language"):
            validate_checkpoint_metadata(checkpoint)

    def test_rejects_invalid_attention_configuration(self):
        checkpoint = valid_checkpoint()
        checkpoint["architecture"]["width"] = 255
        with self.assertRaisesRegex(ValueError, "divisible"):
            validate_checkpoint_metadata(checkpoint)

    def test_rejects_invalid_dropout_and_causal_flags(self):
        for key, value in (("dropout", 1.0), ("causal", "yes")):
            checkpoint = valid_checkpoint()
            checkpoint["architecture"][key] = value
            with self.assertRaises(ValueError):
                validate_checkpoint_metadata(checkpoint)


if __name__ == "__main__":
    unittest.main()
