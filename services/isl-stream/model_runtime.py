"""Fail-closed loader for an explicitly configured, language-specific checkpoint.

A loaded research checkpoint is not sufficient to enable live recognition:
camera preprocessing, calibrated confidence, sentence translation and the
streaming adapter must be implemented and evaluated separately.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

SUPPORTED_LANGUAGES = {"ISL", "ASL", "BSL"}


class ModelRuntime:
    def __init__(self, checkpoint_path: str | None = None, expected_language: str | None = None):
        self.checkpoint_path = checkpoint_path if checkpoint_path is not None else os.getenv(
            "SIGNBRIDGE_MODEL_CHECKPOINT", ""
        )
        self.expected_language = expected_language if expected_language is not None else os.getenv(
            "SIGNBRIDGE_MODEL_LANGUAGE", "ASL"
        )
        if self.expected_language not in SUPPORTED_LANGUAGES:
            raise ValueError("SIGNBRIDGE_MODEL_LANGUAGE must be ASL, ISL, or BSL")
        self._model = None
        self._vocabulary = None
        self._checkpoint: dict[str, Any] | None = None
        self._error: str | None = None

    def load(self) -> bool:
        """Load the configured checkpoint once; never accept a path from a client."""
        if self._model is not None:
            return True
        if not self.checkpoint_path:
            self._error = "model_not_configured"
            return False

        path = Path(self.checkpoint_path).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() not in {".pt", ".pth"}:
            self._error = "checkpoint_file_missing_or_invalid"
            return False

        try:
            import torch
            try:
                from .training.inference import load_model
            except ImportError:  # Running with services/isl-stream as working directory.
                from training.inference import load_model

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, vocabulary, checkpoint = load_model(path, device)
            actual_language = checkpoint.get("sign_language")
            if actual_language != self.expected_language:
                self._error = "checkpoint_language_mismatch"
                return False
        except Exception as exc:
            # Do not expose filesystem details, stack traces or checkpoint contents.
            self._error = "checkpoint_load_failed"
            return False

        self._model = model
        self._vocabulary = vocabulary
        self._checkpoint = checkpoint
        self._error = None
        return True

    def status(self, *, attempt_load: bool = False) -> dict[str, Any]:
        if attempt_load:
            self.load()
        checkpoint_loaded = self._model is not None
        return {
            "service": "signbridge-sign-stream",
            "status": (
                "checkpoint_loaded_streaming_not_integrated"
                if checkpoint_loaded
                else (self._error or "model_not_configured")
            ),
            "model_loaded": checkpoint_loaded,
            "checkpoint_language": (
                self._checkpoint.get("sign_language") if self._checkpoint else None
            ),
            "expected_language": self.expected_language,
            "recognition_enabled": False,
            "translation_enabled": False,
            "message": (
                "A language-matched research checkpoint is loaded, but live camera "
                "recognition and sentence translation are not integrated."
                if checkpoint_loaded
                else "No usable language-matched recognition checkpoint is loaded."
            ),
        }
