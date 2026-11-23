import unittest
from types import SimpleNamespace

import numpy as np

from training.extract_landmarks import (
    FEATURE_DIM,
    pack_landmarks,
    safe_clip_filename,
    sampling_stride,
    validate_signer_splits,
)


def points(count, **values):
    defaults = {"x": 0.1, "y": 0.2, "z": 0.3, "visibility": 0.9, "presence": 0.8}
    defaults.update(values)
    return [SimpleNamespace(**defaults) for _ in range(count)]


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


class ExtractorPrimitiveTests(unittest.TestCase):
    def test_sampling_stride_never_exceeds_requested_rate(self):
        self.assertEqual(sampling_stride(60, 30), 2)
        self.assertEqual(sampling_stride(59.94, 30), 2)
        self.assertEqual(sampling_stride(24, 30), 1)

    def test_sampling_stride_rejects_invalid_rates(self):
        for source, target in ((0, 30), (30, 0), (float("nan"), 30)):
            with self.subTest(source=source, target=target), self.assertRaises(ValueError):
                sampling_stride(source, target)

    def test_clip_filename_is_safe_and_nonempty(self):
        self.assertEqual(safe_clip_filename("../clip 01"), "___clip_01")
        with self.assertRaises(ValueError):
            safe_clip_filename(" .. ")

    def test_pack_landmarks_has_fixed_finite_shape_and_presence_flags(self):
        results = SimpleNamespace(
            pose_landmarks=SimpleNamespace(landmark=points(33)),
            face_landmarks=SimpleNamespace(landmark=points(468)),
            left_hand_landmarks=SimpleNamespace(landmark=points(21)),
            right_hand_landmarks=None,
        )
        vector = pack_landmarks(results)
        self.assertEqual(vector.shape, (FEATURE_DIM,))
        self.assertTrue(np.isfinite(vector).all())
        np.testing.assert_array_equal(vector[-4:], [1, 1, 1, 0])

    def test_missing_landmarks_are_zero_filled(self):
        results = SimpleNamespace(
            pose_landmarks=None, face_landmarks=None,
            left_hand_landmarks=None, right_hand_landmarks=None,
        )
        vector = pack_landmarks(results)
        self.assertEqual(vector.shape, (FEATURE_DIM,))
        np.testing.assert_array_equal(vector, np.zeros(FEATURE_DIM, dtype=np.float32))

    def test_rejects_unexpected_landmark_count(self):
        results = SimpleNamespace(
            pose_landmarks=SimpleNamespace(landmark=points(2)),
            face_landmarks=None, left_hand_landmarks=None, right_hand_landmarks=None,
        )
        with self.assertRaisesRegex(ValueError, "expected 33 landmarks"):
            pack_landmarks(results)


if __name__ == "__main__":
    unittest.main()
