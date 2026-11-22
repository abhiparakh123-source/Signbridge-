"""Load an experimental checkpoint and decode one extracted landmark sequence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import LandmarkCTCModel
from checkpoint_validation import validate_checkpoint_metadata


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


def ctc_greedy_decode(ids, vocabulary):
    reverse = {index: token for token, index in vocabulary.items()}
    result = []
    previous = None
    for raw in ids:
        token_id = int(raw)
        if token_id != previous and token_id != 0:
            if token_id not in reverse:
                raise ValueError("model emitted a token absent from its vocabulary")
            result.append(reverse[token_id])
        previous = token_id
    return result


def predict(features: np.ndarray, model, device: torch.device, vocabulary):
    if features.ndim != 2 or features.shape[0] < 1 or not np.isfinite(features).all():
        raise ValueError("features must be a finite [frames,feature_dim] array")
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
    gloss = predict(features, model, device, vocabulary)
    print(json.dumps({
        "sign_language": args.language,
        "gloss_tokens": gloss,
        "text": " ".join(gloss),
        "status": "experimental_gloss_output_not_sentence_translation",
        "warning": "Research-only output. No calibrated confidence or natural-language translation is provided.",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
