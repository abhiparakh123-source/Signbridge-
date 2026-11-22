"""Unit tests for fixed-width landmark packing (no camera or MediaPipe runtime needed)."""
from types import SimpleNamespace
import unittest

import numpy as np

from training.extract_landmarks import FEATURE_DIM, pack_landmarks


def points(count, **values):
    return SimpleNamespace(landmark=[
        SimpleNamespace(x=values.get("x", 0.1), y=0.2, z=0.3,
                        visibility=0.9, presence=0.8)
        for _ in range(count)
    ])


class LandmarkPackingTests(unittest.TestCase):
    def test_empty_frame_has_fixed_width_and_zero_presence(self):
        result = SimpleNamespace(
            pose_landmarks=None, face_landmarks=None,
            left_hand_landmarks=None, right_hand_landmarks=None,
        )
        vector = pack_landmarks(result)
        self.assertEqual(vector.shape, (FEATURE_DIM,))
        self.assertEqual(FEATURE_DIM, 1699)
        self.assertTrue(np.isfinite(vector).all())
        self.assertEqual(float(vector[-4:].sum()), 0.0)

    def test_present_parts_set_presence_flags_and_coordinates(self):
        result = SimpleNamespace(
            pose_landmarks=points(33, x=0.4),
            face_landmarks=points(468),
            left_hand_landmarks=points(21),
            right_hand_landmarks=None,
        )
        vector = pack_landmarks(result)
        self.assertEqual(vector.shape, (1699,))
        self.assertEqual(vector[-4:].tolist(), [1.0, 1.0, 1.0, 0.0])
        self.assertAlmostEqual(float(vector[0]), 0.4)
        self.assertTrue(np.isfinite(vector).all())

    def test_wrong_landmark_count_is_rejected(self):
        result = SimpleNamespace(
            pose_landmarks=points(32), face_landmarks=None,
            left_hand_landmarks=None, right_hand_landmarks=None,
        )
        with self.assertRaises(ValueError):
            pack_landmarks(result)


if __name__ == "__main__":
    unittest.main()
