"""Tests for dependency-free streaming latency metrics."""
import unittest

from latency_metrics import summarize_latency


class LatencyMetricsTests(unittest.TestCase):
    def test_reports_stage_percentiles_and_frame_loss(self):
        result = summarize_latency([
            {"captured_at_ms": 100, "first_provisional_at_ms": 130,
             "stable_at_ms": 180, "translated_at_ms": 210,
             "captured_frames": 100, "dropped_frames": 5},
            {"captured_at_ms": 200, "first_provisional_at_ms": 240,
             "stable_at_ms": 300, "translated_at_ms": 330,
             "captured_frames": 200, "dropped_frames": 10},
        ])
        self.assertEqual(result["first_provisional_latency_ms"]["p50"], 30)
        self.assertEqual(result["first_provisional_latency_ms"]["p95"], 40)
        self.assertEqual(result["stable_latency_ms"]["p50"], 80)
        self.assertEqual(result["translation_latency_ms"]["p95"], 130)
        self.assertEqual(result["dropped_frames"], 15)
        self.assertEqual(result["dropped_frame_percent"], 5.0)

    def test_empty_input_has_null_percentiles(self):
        result = summarize_latency([])
        self.assertIsNone(result["first_provisional_latency_ms"]["p50"])
        self.assertEqual(result["first_provisional_latency_ms"]["samples"], 0)
        self.assertIsNone(result["dropped_frame_percent"])

    def test_rejects_out_of_order_pipeline_timestamps(self):
        with self.assertRaisesRegex(ValueError, "must not precede"):
            summarize_latency([{
                "captured_at_ms": 10,
                "first_provisional_at_ms": 30,
                "stable_at_ms": 25,
            }])

    def test_rejects_non_finite_and_boolean_timestamps(self):
        for value in (float("nan"), float("inf"), True, -1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    summarize_latency([{"captured_at_ms": value}])

    def test_rejects_frame_loss_above_captured_frames(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            summarize_latency([{
                "captured_at_ms": 0,
                "captured_frames": 3,
                "dropped_frames": 4,
            }])

    def test_rejects_fractional_frame_counts(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            summarize_latency([{
                "captured_at_ms": 0,
                "captured_frames": 2.5,
            }])

    def test_preserves_large_integer_frame_counters_exactly(self):
        result = summarize_latency([{
            "captured_at_ms": 0,
            "captured_frames": 2**60,
            "dropped_frames": 2**59,
        }])
        self.assertEqual(result["captured_frames"], 2**60)
        self.assertEqual(result["dropped_frames"], 2**59)
        self.assertEqual(result["dropped_frame_percent"], 50.0)

    def test_rejects_negative_frame_counters(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            summarize_latency([{
                "captured_at_ms": 0,
                "captured_frames": -1,
            }])


if __name__ == "__main__":
    unittest.main()
