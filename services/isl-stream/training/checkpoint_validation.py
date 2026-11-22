"""Lightweight, dependency-free validation for experimental model metadata."""
from __future__ import annotations


def validate_checkpoint_metadata(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    if checkpoint.get("model_type") != "landmark_ctc_transformer_baseline":
        raise ValueError("unsupported or unverified model checkpoint type")

    vocabulary = checkpoint.get("vocabulary")
    if not isinstance(vocabulary, dict) or vocabulary.get("<blank>") != 0:
        raise ValueError("checkpoint vocabulary is invalid")
    token_ids = list(vocabulary.values())
    if (
        any(not isinstance(token, str) or not token.strip() for token in vocabulary)
        or any(isinstance(index, bool) or not isinstance(index, int) for index in token_ids)
        or sorted(token_ids) != list(range(len(vocabulary)))
    ):
        raise ValueError("checkpoint vocabulary IDs must be unique, contiguous integers")

    feature_dim = checkpoint.get("feature_dim")
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
        raise ValueError("checkpoint feature_dim must be a positive integer")

    language = checkpoint.get("sign_language")
    if language not in {"ISL", "ASL", "BSL"}:
        raise ValueError("checkpoint sign_language metadata is invalid")

    architecture = checkpoint.get("architecture", {})
    if not isinstance(architecture, dict):
        raise ValueError("checkpoint architecture metadata is invalid")
    architecture = dict(architecture)
    for key, default in (("width", 256), ("heads", 4), ("layers", 4)):
        value = architecture.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"checkpoint architecture {key} must be a positive integer")
        architecture[key] = value
    if architecture["width"] % architecture["heads"] != 0:
        raise ValueError("checkpoint architecture width must be divisible by heads")
    dropout = architecture.get("dropout", 0.1)
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ValueError("checkpoint architecture dropout must be in [0, 1)")
    architecture["dropout"] = float(dropout)

    return vocabulary, feature_dim, language, architecture
