import unittest

from training.extract_landmarks import validate_signer_splits


class ExtractorSplitValidationTests(unittest.TestCase):
    def test_accepts_signer_disjoint_splits(self):
        validate_signer_splits([
            {"signer_id": "s1", "split": "train"},
            {"signer_id": "s2", "split": "validation"},
            {"signer_id": "s3", "split": "test"},
        ])

    def test_rejects_signer_across_train_and_test(self):
        with self.assertRaisesRegex(ValueError, "s1: train, test"):
            validate_signer_splits([
                {"signer_id": "s1", "split": "train"},
                {"signer_id": "s1", "split": "test"},
            ])

    def test_rejects_missing_signer(self):
        with self.assertRaisesRegex(ValueError, "signer_id is required"):
            validate_signer_splits([{"signer_id": "", "split": "train"}])

    def test_rejects_invalid_split(self):
        with self.assertRaisesRegex(ValueError, "split must be"):
            validate_signer_splits([{"signer_id": "s1", "split": "holdout"}])


if __name__ == "__main__":
    unittest.main()
