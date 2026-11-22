"""Train a single-language CTC baseline from consented landmark .npy files.

Manifest columns: clip_id, signer_id, sign_language, feature_path, gloss,
consent, split. Features must be finite float arrays shaped [frames, features].
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from model import LandmarkCTCModel


def load_manifest(path: Path, root: Path, language: str):
    required = {"clip_id", "signer_id", "sign_language", "feature_path", "gloss", "consent", "split"}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("manifest is missing required columns: " + ", ".join(sorted(required - set(reader.fieldnames or []))))
        rows = list(reader)
    if not rows:
        raise ValueError("manifest is empty")
    signer_splits = {}
    clip_ids = set()
    checked = []
    for row in rows:
        clip_id = row["clip_id"].strip()
        signer = row["signer_id"].strip()
        split = row["split"].strip().lower()
        if not clip_id or clip_id in clip_ids:
            raise ValueError("clip IDs must be non-empty and unique")
        clip_ids.add(clip_id)
        if not signer or split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid signer or split for clip {clip_id}")
        if row["consent"].strip().lower() not in {"yes", "true", "1"}:
            raise ValueError(f"clip {clip_id} has no explicit consent")
        if row["sign_language"].strip().upper() != language:
            raise ValueError(f"manifest must contain only {language} clips for this run")
        previous = signer_splits.setdefault(signer, split)
        if previous != split:
            raise ValueError(f"signer {signer} appears in multiple splits")
        feature_path = (root / row["feature_path"]).resolve()
        if not feature_path.is_relative_to(root.resolve()):
            raise ValueError(f"feature path escapes dataset root for clip {clip_id}")
        if not feature_path.is_file() or feature_path.suffix != ".npy":
            raise ValueError(f"missing .npy feature file for clip {clip_id}")
        gloss = row["gloss"].strip()
        if not gloss:
            raise ValueError(f"empty gloss for clip {clip_id}")
        checked.append({**row, "_path": feature_path, "_split": split, "_gloss": gloss})
    if not {"train", "validation", "test"}.issubset({r["_split"] for r in checked}):
        raise ValueError("train, validation and test splits are all required")
    return checked


class LandmarkDataset(Dataset):
    def __init__(self, rows, vocabulary):
        self.rows = rows
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        values = np.load(row["_path"], allow_pickle=False)
        if values.ndim != 2 or values.shape[0] < 1 or not np.isfinite(values).all():
            raise ValueError(f"invalid feature array for clip {row['clip_id']}")
        gloss_tokens = row["_gloss"].split()
        if not gloss_tokens:
            raise ValueError(f"empty token sequence for clip {row['clip_id']}")
        unknown = set(gloss_tokens) - self.vocabulary.keys()
        if unknown:
            raise ValueError(f"clip {row['clip_id']} contains gloss tokens absent from training vocabulary: {sorted(unknown)}")
        minimum_frames = len(gloss_tokens) + sum(
            left == right for left, right in zip(gloss_tokens, gloss_tokens[1:])
        )
        if minimum_frames > values.shape[0]:
            raise ValueError(f"clip {row['clip_id']} has too few frames for a valid CTC alignment")
        tokens = [self.vocabulary[token] for token in gloss_tokens]
        return torch.from_numpy(values.astype(np.float32)), torch.tensor(tokens, dtype=torch.long)


def collate(batch):
    sequences, targets = zip(*batch)
    lengths = torch.tensor([len(item) for item in sequences], dtype=torch.long)
    features = pad_sequence(sequences, batch_first=True)
    target_lengths = torch.tensor([len(item) for item in targets], dtype=torch.long)
    return features, lengths, torch.cat(targets), target_lengths


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for row, ref_token in enumerate(reference, start=1):
        current = [row]
        for column, hyp_token in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (ref_token != hyp_token),
            ))
        previous = current
    return previous[-1]


def greedy_ctc_decode(frame_ids, blank_id=0):
    decoded = []
    previous = None
    for token in frame_ids:
        token = int(token)
        if token != previous and token != blank_id:
            decoded.append(token)
        previous = token
    return decoded


def evaluate(model, loader, loss_fn, device, reverse_vocabulary):
    model.eval()
    total_loss = 0.0
    count = 0
    word_errors = word_total = char_errors = char_total = 0
    with torch.no_grad():
        for features, lengths, targets, target_lengths in loader:
            features, lengths = features.to(device), lengths.to(device)
            targets, target_lengths = targets.to(device), target_lengths.to(device)
            log_probs, output_lengths = model(features, lengths)
            loss = loss_fn(log_probs, targets, output_lengths, target_lengths)
            total_loss += float(loss) * features.shape[0]
            count += features.shape[0]
            predictions = log_probs.argmax(dim=-1).transpose(0, 1)
            offset = 0
            for index, target_length in enumerate(target_lengths.tolist()):
                reference_ids = targets[offset:offset + target_length].tolist()
                offset += target_length
                predicted_ids = greedy_ctc_decode(
                    predictions[index, :int(output_lengths[index])].tolist()
                )
                reference = [reverse_vocabulary[token] for token in reference_ids]
                hypothesis = [reverse_vocabulary[token] for token in predicted_ids]
                word_errors += edit_distance(reference, hypothesis)
                word_total += len(reference)
                ref_text, hyp_text = " ".join(reference), " ".join(hypothesis)
                char_errors += edit_distance(list(ref_text), list(hyp_text))
                char_total += len(ref_text)
    return {
        "ctc_loss": total_loss / max(count, 1),
        "gloss_wer": word_errors / max(word_total, 1),
        "gloss_cer": char_errors / max(char_total, 1),
        "clips": count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--language", choices=["ISL", "ASL", "BSL"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_manifest(args.manifest, args.data_root, args.language)
    train_rows = [r for r in rows if r["_split"] == "train"]
    val_rows = [r for r in rows if r["_split"] == "validation"]
    test_rows = [r for r in rows if r["_split"] == "test"]
    vocabulary_tokens = sorted({token for row in train_rows for token in row["_gloss"].split()})
    if not vocabulary_tokens:
        raise ValueError("training split has no gloss vocabulary")
    vocabulary = {"<blank>": 0, **{token: i + 1 for i, token in enumerate(vocabulary_tokens)}}
    feature_dim = None
    for row in train_rows:
        array = np.load(row["_path"], mmap_mode="r", allow_pickle=False)
        if array.ndim != 2 or array.shape[0] < 1:
            raise ValueError(f"invalid feature shape for {row['clip_id']}")
        feature_dim = feature_dim or int(array.shape[1])
        if int(array.shape[1]) != feature_dim:
            raise ValueError("feature dimensions differ across clips")
        unknown = set(row["_gloss"].split()) - vocabulary.keys()
        if unknown:
            raise ValueError("training vocabulary construction failed")
    if feature_dim is None:
        raise ValueError("no training feature arrays found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LandmarkCTCModel(feature_dim, len(vocabulary)).to(device)
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    train_loader = DataLoader(LandmarkDataset(train_rows, vocabulary), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(LandmarkDataset(val_rows, vocabulary), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(LandmarkDataset(test_rows, vocabulary), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, device)
        reverse_vocabulary = {index: token for token, index in vocabulary.items()}
        validation_metrics = evaluate(model, val_loader, loss_fn, device, reverse_vocabulary)
        val_loss = validation_metrics["ctc_loss"]
        history.append({"epoch": epoch, "train_ctc_loss": train_loss, "validation": validation_metrics})
        print(json.dumps(history[-1]))
        if val_loss < best:
            best = val_loss
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "feature_dim": feature_dim,
                "vocabulary": vocabulary,
                "sign_language": args.language,
                "model_type": "landmark_ctc_transformer_baseline",
                "architecture": {"width": 256, "heads": 4, "layers": 4, "dropout": 0.1},
                "seed": args.seed,
            }, args.output)
    checkpoint = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate(model, test_loader, loss_fn, device, reverse_vocabulary)
    report = {
        "language": args.language,
        "device": str(device),
        "test_metrics": test_metrics,
        "history": history,
        "warning": "Gloss WER/CER measure this experimental gloss baseline only, not sentence translation quality or real-world sign comprehension.",
    }
    args.output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"test_metrics": test_metrics, "checkpoint": str(args.output)}))


if __name__ == "__main__":
    main()
