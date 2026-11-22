"""Tests for chunk-safe CTC greedy decoding without ML dependencies."""
import unittest

from training.ctc_decode import decode_frame_ids_incremental


class IncrementalCTCDecodeTests(unittest.TestCase):
    VOCABULARY = {"<blank>": 0, "HELLO": 1, "WORLD": 2}

    def test_decodes_repeated_tokens_with_blank_separation(self):
        tokens, previous = decode_frame_ids_incremental(
            [1, 1, 0, 1, 2, 2, 0], self.VOCABULARY
        )
        self.assertEqual(tokens, ["HELLO", "HELLO", "WORLD"])
        self.assertEqual(previous, 0)

    def test_preserves_repeat_state_across_chunk_boundary(self):
        first, previous = decode_frame_ids_incremental([1, 1], self.VOCABULARY)
        second, previous = decode_frame_ids_incremental(
            [1, 0, 1, 2], self.VOCABULARY, previous
        )
        self.assertEqual(first + second, ["HELLO", "HELLO", "WORLD"])
        self.assertEqual(previous, 2)

    def test_blank_resets_repeat_state_between_chunks(self):
        first, previous = decode_frame_ids_incremental([1, 0], self.VOCABULARY)
        second, previous = decode_frame_ids_incremental([1], self.VOCABULARY, previous)
        self.assertEqual(first + second, ["HELLO", "HELLO"])
        self.assertEqual(previous, 1)

    def test_rejects_unknown_token_ids(self):
        with self.assertRaisesRegex(ValueError, "absent from vocabulary"):
            decode_frame_ids_incremental([3], self.VOCABULARY)


if __name__ == "__main__":
    unittest.main()
