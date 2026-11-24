"""Load an experimental checkpoint and decode one extracted landmark sequence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:  # Package import for tests and service integration.
    from .model import LandmarkCTCModel
    from .checkpoint_validation import validate_checkpoint_metadata
    from .ctc_decode import decode_frame_ids_incremental
    from .chunking import iter_bounded_windows
except ImportError:  # Direct script execution from the training directory.
    from model import LandmarkCTCModel
    from checkpoint_validation import validate_checkpoint_metadata
    from ctc_decode import decode_frame_ids_incremental
    from chunking import iter_bounded_windows


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    vocabulary, feature_dim, language, architecture = validate_checkpoint_metadata(checkpoint)
    model = LandmarkCTCModel(
        feature_dim=feature_dim,
        vocabulary_size=len(vocabulary),
        width=int(architecture.get("width", 256)),
        heads=int(architecture.get("heads", 4)),
        layers=int(architecture.get("layers", 4)),
        dropout=float(architecture.get("dropout", 0.1)),
        causal=architecture.get("causal", False),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, vocabulary, checkpoint


def normalize_landmark_features(features: np.ndarray, normalization: dict | None) -> np.ndarray:
    """Apply the exact training-split statistics saved in a checkpoint."""
    values = np.asarray(features, dtype=np.float32)
    if normalization is None:
        return values
    if normalization.get("method") != "coordinate_zscore_observed_landmarks":
        raise ValueError("unsupported checkpoint normalization method")
    if normalization.get("missing_landmark") != "preserve_zero_triple":
        raise ValueError("unsupported missing-landmark normalization policy")
    mean = np.asarray(normalization.get("mean"), dtype=np.float32)
    scale = np.asarray(normalization.get("scale"), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 384 or mean.shape != (3,) or scale.shape != (3,):
        raise ValueError("landmark normalization metadata or feature shape is invalid")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("landmark normalization statistics must be finite and positive")
    points = values.reshape(values.shape[0], 128, 3).copy()
    present = np.any(points != 0.0, axis=-1)
    points[present] = (points[present] - mean) / scale
    points[~present] = 0.0
    return points.reshape(values.shape[0], 384)


def ctc_greedy_decode(ids, vocabulary):
    tokens, _ = decode_frame_ids_incremental(ids, vocabulary)
    return tokens


def iter_causal_predictions(chunks, model, device: torch.device, vocabulary, normalization=None):
    """Yield newly decoded gloss tokens as chunks arrive.

    This research path recomputes the entire causal prefix for each chunk. It
    demonstrates prefix-stable emissions, not efficient cached streaming.
    """
    if not getattr(model, "causal", False):
        raise ValueError("incremental prefix emission requires a causal checkpoint")
    accumulated = None
    processed_frames = 0
    previous_frame_id = None
    feature_dim = None

    for chunk in chunks:
        values = np.asarray(chunk)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("each chunk must be a finite [frames,feature_dim] array")
        if values.shape[0] == 0:
            continue
        if feature_dim is None:
            feature_dim = values.shape[1]
        if values.shape[1] != feature_dim:
            raise ValueError("feature dimensions changed between chunks")
        values = normalize_landmark_features(values.astype(np.float32, copy=False), normalization)
        accumulated = values if accumulated is None else np.concatenate((accumulated, values), axis=0)
        tensor = torch.from_numpy(accumulated).unsqueeze(0).to(device)
        lengths = torch.tensor([len(accumulated)], dtype=torch.long, device=device)
        with torch.no_grad():
            log_probs, output_lengths = model(tensor, lengths)
            new_ids = (
                log_probs[processed_frames:int(output_lengths[0]), 0]
                .argmax(dim=-1).cpu().tolist()
            )
        emitted, previous_frame_id = decode_frame_ids_incremental(
            new_ids, vocabulary, previous_frame_id
        )
        processed_frames = int(output_lengths[0])
        yield emitted



def iter_bounded_causal_predictions(
    chunks,
    model,
    device: torch.device,
    vocabulary,
    *,
    max_frames: int,
    overlap_frames: int = 0,
    normalization=None,
):
    """Yield newly decoded gloss tokens from bounded causal windows.

    Overlap frames provide context but are not decoded twice. CTC repeat state
    is preserved across emitted regions. Each window is evaluated independently:
    this bounds per-call memory, but is not cached streaming inference and
    window-local positional encodings can still affect boundary predictions.
    """
    if not getattr(model, "causal", False):
        raise ValueError("bounded incremental inference requires a causal checkpoint")

    previous_frame_id = None
    emitted_until = 0
    for start, end, window in iter_bounded_windows(
        chunks, max_frames=max_frames, overlap_frames=overlap_frames
    ):
        window = normalize_landmark_features(window, normalization)
        tensor = torch.from_numpy(window).unsqueeze(0).to(device)
        lengths = torch.tensor([len(window)], dtype=torch.long, device=device)
        with torch.no_grad():
            log_probs, output_lengths = model(tensor, lengths)
            valid_length = int(output_lengths[0])
            if valid_length < 1 or valid_length > len(window):
                raise ValueError("model returned an invalid output length")
            first_new = max(0, emitted_until - start)
            if first_new >= valid_length:
                emitted_until = max(emitted_until, end)
                continue
            new_ids = (
                log_probs[first_new:valid_length, 0]
                .argmax(dim=-1)
                .cpu()
                .tolist()
            )
        emitted, previous_frame_id = decode_frame_ids_incremental(
            new_ids, vocabulary, previous_frame_id
        )
        emitted_until = max(emitted_until, start + valid_length)
        yield emitted




def predict(features: np.ndarray, model, device: torch.device, vocabulary, normalization=None):
    if features.ndim != 2 or features.shape[0] < 1 or not np.isfinite(features).all():
        raise ValueError("features must be a finite [frames,feature_dim] array")
    features = normalize_landmark_features(features, normalization)
    tensor = torch.from_numpy(features.astype(np.float32, copy=False)).unsqueeze(0).to(device)
    lengths = torch.tensor([features.shape[0]], dtype=torch.long, device=device)
    with torch.no_grad():
        log_probs, output_lengths = model(tensor, lengths)
        frame_ids = log_probs[:int(output_lengths[0]), 0].argmax(dim=-1).cpu().tolist()
    return ctc_greedy_decode(frame_ids, vocabulary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True, help="one .npy sequence from extract_landmarks.py")
    parser.add_argument("--language", choices=["ISL", "ASL", "BSL"], required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocabulary, checkpoint = load_model(args.checkpoint, device)
    if checkpoint.get("sign_language") != args.language:
        parser.error("checkpoint language does not match requested language")
    features = np.load(args.features, allow_pickle=False)
    if features.ndim != 2 or features.shape[0] < 1:
        parser.error("features must be a non-empty [frames,feature_dim] array")
    if features.shape[1] != int(checkpoint["feature_dim"]):
        parser.error("feature dimensions do not match this checkpoint")
    gloss = predict(features, model, device, vocabulary, checkpoint.get("normalization"))
    print(json.dumps({
        "sign_language": args.language,
        "gloss_tokens": gloss,
        "text": " ".join(gloss),
        "status": "experimental_gloss_output_not_sentence_translation",
        "warning": "Research-only output. No calibrated confidence or natural-language translation is provided.",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
