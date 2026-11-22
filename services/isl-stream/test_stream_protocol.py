import base64
import unittest

from stream_protocol import (
    ProtocolError,
    validate_frame_message,
    validate_hypothesis,
)


class StreamProtocolTests(unittest.TestCase):
    def frame(self, **overrides):
        data = {
            "type": "frame",
            "sign_language": "ISL",
            "sequence": 0,
            "timestamp_ms": 10,
            "mime_type": "image/jpeg",
            "frame_b64": base64.b64encode(b"image").decode("ascii"),
        }
        data.update(overrides)
        return data

    def test_valid_frame_envelope(self):
        result = validate_frame_message(self.frame())
        self.assertEqual(result["frame_bytes"], 5)

    def test_language_is_mandatory_and_specific(self):
        with self.assertRaises(ProtocolError):
            validate_frame_message(self.frame(sign_language="Hindi"))

    def test_timestamps_must_increase(self):
        with self.assertRaises(ProtocolError):
            validate_frame_message(self.frame(timestamp_ms=10), last_timestamp_ms=10)

    def test_sequences_must_increase(self):
        with self.assertRaises(ProtocolError):
            validate_frame_message(self.frame(sequence=4), last_sequence=4)
        with self.assertRaises(ProtocolError):
            validate_frame_message(self.frame(sequence=2), last_sequence=3)

    def test_invalid_base64_is_rejected(self):
        with self.assertRaises(ProtocolError):
            validate_frame_message(self.frame(frame_b64="%%%"))

    def test_provisional_text_cannot_be_spoken(self):
        result = validate_hypothesis({
            "type": "hypothesis", "sign_language": "ISL",
            "state": "provisional", "text": "hello", "confidence": 0.99,
        })
        self.assertFalse(result["speak_allowed"])

    def test_uncertain_text_cannot_be_spoken(self):
        result = validate_hypothesis({
            "type": "hypothesis", "sign_language": "ASL",
            "state": "uncertain", "text": "hello", "confidence": 0.99,
        })
        self.assertFalse(result["speak_allowed"])

    def test_stable_high_confidence_text_can_be_spoken(self):
        result = validate_hypothesis({
            "type": "hypothesis", "sign_language": "BSL",
            "state": "stable", "text": "hello", "confidence": 0.95,
        })
        self.assertTrue(result["speak_allowed"])


    def test_unhashable_language_and_mime_are_rejected_cleanly(self):
        with self.assertRaises(ProtocolError):
            validate_frame_message(self.frame(sign_language=[]))
        with self.assertRaises(ProtocolError):
            validate_frame_message(self.frame(mime_type=[]))

    def test_non_finite_confidence_values_are_rejected(self):
        for confidence in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(confidence=confidence), self.assertRaises(ProtocolError):
                validate_hypothesis({
                    "type": "hypothesis", "sign_language": "ISL",
                    "state": "stable", "text": "hello", "confidence": confidence,
                })

    def test_invalid_speech_threshold_is_rejected(self):
        message = {
            "type": "hypothesis", "sign_language": "ISL",
            "state": "stable", "text": "hello", "confidence": 0.99,
        }
        for threshold in (float("nan"), float("inf"), True, "0.9"):
            with self.subTest(threshold=threshold), self.assertRaises(ProtocolError):
                validate_hypothesis(message, min_speak_confidence=threshold)

    def test_unhashable_hypothesis_state_is_rejected_cleanly(self):
        with self.assertRaises(ProtocolError):
            validate_hypothesis({
                "type": "hypothesis", "sign_language": "ISL",
                "state": [], "text": "hello", "confidence": 0.99,
            })

    def test_oversized_hypothesis_text_is_rejected(self):
        with self.assertRaisesRegex(ProtocolError, "4096-character"):
            validate_hypothesis({
                "type": "hypothesis", "sign_language": "ISL",
                "state": "stable", "text": "x" * 4097, "confidence": 0.99,
            })

    def test_stable_low_confidence_text_is_not_spoken(self):
        result = validate_hypothesis({
            "type": "hypothesis", "sign_language": "ISL",
            "state": "stable", "text": "hello", "confidence": 0.55,
        })
        self.assertFalse(result["speak_allowed"])


if __name__ == "__main__":
    unittest.main()
