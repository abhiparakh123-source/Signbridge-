import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from training.extract_landmarks import (
    FEATURE_DIM,
    frame_window,
    build_feature_manifest_row,
    pack_landmarks,
    safe_clip_filename,
    sha256_file,
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


class ExtractorHashTests(unittest.TestCase):
    def test_sha256_file_reads_in_small_chunks_and_matches_standard_digest(self):
        payload = b"signbridge-landmark-feature-data" * 100
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npy"
            path.write_bytes(payload)
            self.assertEqual(
                sha256_file(path, chunk_size=17),
                hashlib.sha256(payload).hexdigest(),
            )

    def test_sha256_rejects_non_positive_chunk_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.bin"
            path.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "chunk_size must be positive"):
                sha256_file(path, chunk_size=0)


class ExtractorProvenanceTests(unittest.TestCase):
    def test_feature_manifest_preserves_asl_source_provenance_and_hash(self):
        source = {
            "clip_id": "corpus__clip-1",
            "signer_id": "signer-1",
            "sign_language": "asl",
            "label": "HELLO WORLD",
            "consent": "yes",
            "split": "validation",
            "video_path": "clips/clip-1.mp4",
            "source_dataset": "How2Sign",
            "source_clip_id": "clip-1",
            "start_ms": "1200",
            "end_ms": "2400",
            "license_id": "CC-BY-NC-4.0",
            "attribution": "Dataset authors",
            "source_url": "https://example.org/source",
        }
        result = build_feature_manifest_row(
            source, __import__("pathlib").Path("features/clip-1.npy"),
            {"sha256": "abc123"},
        )
        self.assertEqual(result["sign_language"], "ASL")
        self.assertEqual(result["source_dataset"], "How2Sign")
        self.assertEqual(result["source_clip_id"], "clip-1")
        self.assertEqual(result["start_ms"], "1200")
        self.assertEqual(result["end_ms"], "2400")
        self.assertEqual(result["license_id"], "CC-BY-NC-4.0")
        self.assertEqual(result["source_url"], "https://example.org/source")
        self.assertEqual(result["feature_sha256"], "abc123")

    def test_family_manifest_without_source_fields_gets_safe_defaults(self):
        source = {
            "clip_id": "family-clip", "signer_id": "family-signer",
            "sign_language": "ASL", "label": "HELLO", "consent": "yes",
            "split": "train", "video_path": "clips/family.mp4",
        }
        result = build_feature_manifest_row(
            source, __import__("pathlib").Path("features/family.npy"),
            {"sha256": "def456"},
        )
        self.assertEqual(result["source_clip_id"], "family-clip")
        self.assertEqual(result["license_id"], "")
        self.assertEqual(result["feature_sha256"], "def456")


class ExtractorTimestampTests(unittest.TestCase):
    def test_timestamp_interval_maps_to_half_open_frame_window(self):
        self.assertEqual(frame_window(30.0, 1000, 2500), (30, 75))

    def test_open_ended_family_clip_starts_at_zero(self):
        self.assertEqual(frame_window(25.0), (0, None))

    def test_timestamp_window_rejects_negative_or_reversed_ranges(self):
        for start, end in ((-1, 1000), (1000, 1000), (2000, 1000)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                frame_window(30.0, start, end)

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
        self.assertEqual(safe_clip_filename("../clip 01"), "clip_01")
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
