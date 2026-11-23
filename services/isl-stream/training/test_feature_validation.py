"""Tests for bounded-memory NumPy feature validation."""
import unittest

import numpy as np

from training.feature_validation import validate_feature_array


class FeatureValidationTests(unittest.TestCase):
    def test_accepts_finite_array_in_multiple_chunks(self):
        values = np.ones((11, 5), dtype=np.float32)
        self.assertEqual(validate_feature_array(values, chunk_rows=3), (11, 5))

    def test_rejects_nan_in_later_chunk(self):
        values = np.ones((11, 5), dtype=np.float32)
        values[8, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "rows 6:9"):
            validate_feature_array(values, chunk_rows=3)

    def test_rejects_infinite_values(self):
        values = np.ones((2, 3), dtype=np.float32)
        values[1, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_feature_array(values)

    def test_rejects_invalid_shape_and_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "2-D"):
            validate_feature_array(np.ones(5))
        with self.assertRaisesRegex(ValueError, "chunk_rows must be positive"):
            validate_feature_array(np.ones((2, 3)), chunk_rows=0)


if __name__ == "__main__":
    unittest.main()
