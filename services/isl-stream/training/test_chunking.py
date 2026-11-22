"""Dependency-light tests for bounded landmark windowing."""
import unittest

import numpy as np

from training.chunking import iter_bounded_windows


class BoundedWindowTests(unittest.TestCase):
    def test_windows_are_bounded_and_keep_requested_overlap(self):
        frames = np.arange(20, dtype=np.float32).reshape(10, 2)
        windows = list(iter_bounded_windows([frames], max_frames=6, overlap_frames=2))
        self.assertEqual([(start, end) for start, end, _ in windows], [(0, 6), (4, 10)])
        np.testing.assert_array_equal(windows[1][2][:2], frames[4:6])
        self.assertTrue(all(len(window) <= 6 for _, _, window in windows))

    def test_accepts_arbitrary_chunk_boundaries(self):
        frames = np.arange(30, dtype=np.float32).reshape(15, 2)
        chunks = [frames[:1], frames[1:8], frames[8:9], frames[9:]]
        windows = list(iter_bounded_windows(chunks, max_frames=5, overlap_frames=1))
        rebuilt = np.concatenate([item[2] for item in windows])
        self.assertEqual([item[:2] for item in windows], [(0, 5), (4, 9), (8, 13), (12, 15)])
        np.testing.assert_array_equal(rebuilt[-3:], frames[-3:])

    def test_short_stream_yields_one_partial_window(self):
        frames = np.ones((3, 4), dtype=np.float32)
        windows = list(iter_bounded_windows([frames], max_frames=8, overlap_frames=2))
        self.assertEqual(len(windows), 1)
        self.assertEqual((windows[0][0], windows[0][1]), (0, 3))

    def test_empty_chunks_are_ignored(self):
        frames = np.ones((2, 3), dtype=np.float32)
        windows = list(iter_bounded_windows([np.empty((0, 3)), frames], 4, 0))
        self.assertEqual(len(windows), 1)
        np.testing.assert_array_equal(windows[0][2], frames)

    def test_rejects_invalid_dimensions_and_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "feature dimensions changed"):
            list(iter_bounded_windows([np.ones((2, 3)), np.ones((2, 4))], 5))
        with self.assertRaisesRegex(ValueError, "finite"):
            list(iter_bounded_windows([np.array([[float("nan")]])], 5))

    def test_rejects_invalid_window_settings(self):
        for maximum, overlap in [(0, 0), (4, 4), (4, -1), (True, 0)]:
            with self.subTest(maximum=maximum, overlap=overlap):
                with self.assertRaises(ValueError):
                    list(iter_bounded_windows([], maximum, overlap))


if __name__ == "__main__":
    unittest.main()
