"""Strict, dependency-free message validation for future streaming inference.

This is a protocol boundary only. It does not inspect images or recognise signs.
"""
from __future__ import annotations

import base64
import binascii
import math
from typing import Any

SUPPORTED_LANGUAGES = {"ISL", "ASL", "BSL"}
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_FRAME_B64_CHARS = ((MAX_FRAME_BYTES + 2) // 3) * 4


class ProtocolError(ValueError):
    """A client or model message violates the streaming contract."""


def validate_frame_message(
    message: dict[str, Any], *, last_timestamp_ms: int = -1, last_sequence: int = -1
) -> dict[str, Any]:
    """Validate one timestamped frame envelope before any image decoding."""
    if not isinstance(message, dict) or message.get("type") != "frame":
        raise ProtocolError("message type must be 'frame'")
    language = message.get("sign_language")
    if not isinstance(language, str) or language not in SUPPORTED_LANGUAGES:
        raise ProtocolError("sign_language must be ISL, ASL, or BSL")
    sequence = message.get("sequence")
    timestamp = message.get("timestamp_ms")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ProtocolError("sequence must be a non-negative integer")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise ProtocolError("timestamp_ms must be a non-negative integer")
    if sequence <= last_sequence:
        raise ProtocolError("sequence must increase monotonically")
    if timestamp <= last_timestamp_ms:
        raise ProtocolError("timestamp_ms must increase monotonically")
    mime = message.get("mime_type")
    if not isinstance(mime, str) or mime not in SUPPORTED_IMAGE_TYPES:
        raise ProtocolError("mime_type must be JPEG, PNG, or WebP")
    encoded = message.get("frame_b64")
    if not isinstance(encoded, str) or not encoded:
        raise ProtocolError("frame_b64 is required")
    if len(encoded) > MAX_FRAME_B64_CHARS:
        raise ProtocolError("frame exceeds the 2 MiB decoded size limit")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ProtocolError("frame_b64 is not valid base64") from None
    if not raw or len(raw) > MAX_FRAME_BYTES:
        raise ProtocolError("decoded frame must be between 1 byte and 2 MiB")
    # Check the actual container signature; a client-supplied MIME label alone
    # must not decide which image decoder handles untrusted bytes.
    signatures = {
        "image/jpeg": raw.startswith(b"\xff\xd8\xff"),
        "image/png": raw.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/webp": (
            len(raw) >= 12 and raw.startswith(b"RIFF") and raw[8:12] == b"WEBP"
        ),
    }
    if not signatures[mime]:
        raise ProtocolError("frame bytes do not match the declared image MIME type")
    return {
        "type": "frame",
        "sign_language": language,
        "sequence": sequence,
        "timestamp_ms": timestamp,
        "mime_type": mime,
        "frame_bytes": len(raw),
    }


def validate_hypothesis(
    message: dict[str, Any], *, min_speak_confidence: float = 0.90
) -> dict[str, Any]:
    """Validate model output and gate speech to stable, high-confidence text."""
    if not isinstance(message, dict) or message.get("type") != "hypothesis":
        raise ProtocolError("message type must be 'hypothesis'")
    language = message.get("sign_language")
    if not isinstance(language, str) or language not in SUPPORTED_LANGUAGES:
        raise ProtocolError("sign_language must be ISL, ASL, or BSL")
    state = message.get("state")
    if not isinstance(state, str) or state not in {"provisional", "stable", "uncertain"}:
        raise ProtocolError("state must be provisional, stable, or uncertain")
    text = message.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ProtocolError("text must be a non-empty string")
    if len(text) > 4096:
        raise ProtocolError("text exceeds the 4096-character limit")
    confidence = message.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ProtocolError("confidence must be a number from 0 to 1")
    try:
        confidence_value = float(confidence)
    except (OverflowError, ValueError):
        raise ProtocolError("confidence must be a finite number from 0 to 1") from None
    if not math.isfinite(confidence_value) or not 0 <= confidence_value <= 1:
        raise ProtocolError("confidence must be a finite number from 0 to 1")

    if isinstance(min_speak_confidence, bool) or not isinstance(min_speak_confidence, (int, float)):
        raise ProtocolError("min_speak_confidence must be a finite number between 0 and 1")
    try:
        threshold = float(min_speak_confidence)
    except (OverflowError, ValueError):
        raise ProtocolError("min_speak_confidence must be a finite number between 0 and 1") from None
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ProtocolError("min_speak_confidence must be a finite number between 0 and 1")
    can_speak = state == "stable" and confidence_value >= threshold
    return {
        "type": "hypothesis",
        "sign_language": language,
        "state": state,
        "text": text.strip(),
        "confidence": float(confidence),
        "speak_allowed": can_speak,
    }
