"""Train a single-language CTC baseline from consented landmark .npy files.

Manifest columns: clip_id, signer_id, sign_language, feature_path, gloss,
consent, split. Features must be finite float arrays shaped [frames, features].
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

try:
    from .model import LandmarkCTCModel
    from .feature_validation import validate_feature_array
except ImportError:  # Direct script execution from the training directory.
    from model import LandmarkCTCModel
    from feature_validation import validate_feature_array


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
        # Corpora with an explicit rights-review field must be approved.
        permission_status = row.get("permission_status", "").strip().lower()
        if permission_status and permission_status not in {"approved", "user_attested"}:
            raise ValueError(f"clip {clip_id} source permission is not approved or user-attested")
        row_language = row["sign_language"].strip().upper()
        if row_language not in {"ISL", "ASL", "BSL"}:
            raise ValueError(f"invalid sign language for clip {clip_id}")
        # Extraction can produce one combined feature manifest. A training run
        # intentionally selects exactly one language; skip other valid languages.
        if row_language != language:
            continue
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


class LandmarkNormalizer:
    """Coordinate-wise normalization fitted on training landmarks only.

    Épée encodes an absent landmark as an all-zero (x, y, z) triple. Such
    triples are excluded from the statistics and remain exactly zero after
    transformation, so missingness is not turned into a fabricated position.
    """
    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        if self.mean.shape != (3,) or self.scale.shape != (3,):
            raise ValueError("normalizer expects three coordinate statistics")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all():
            raise ValueError("normalizer statistics must be finite")
        if np.any(self.scale <= 0):
            raise ValueError("normalizer scales must be positive")

    @classmethod
    def fit(cls, rows):
        totals = np.zeros(3, dtype=np.float64)
        squared_totals = np.zeros(3, dtype=np.float64)
        count = 0
        for row in rows:
            values = np.load(row["_path"], mmap_mode="r", allow_pickle=False)
            validate_feature_array(values)
            if values.shape[1] != 384:
                raise ValueError("landmark normalization requires 384-D (128 x 3) features")
            points = np.asarray(values, dtype=np.float64).reshape(-1, 128, 3)
            present = np.any(points != 0.0, axis=-1)
            observed = points[present]
            if observed.size:
                totals += observed.sum(axis=0)
                squared_totals += np.square(observed).sum(axis=0)
                count += observed.shape[0]
        if count == 0:
            raise ValueError("training split contains no present landmarks")
        mean = totals / count
        variance = np.maximum(squared_totals / count - np.square(mean), 1e-8)
        return cls(mean.astype(np.float32), np.sqrt(variance).astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 384:
            raise ValueError("normalizer expects [frames, 384] features")
        points = array.reshape(-1, 128, 3).copy()
        present = np.any(points != 0.0, axis=-1)
        points[present] = (points[present] - self.mean) / self.scale
        # Absent landmarks are deliberately preserved as exact zero triples.
        return points.reshape(-1, 384)


class LandmarkDataset(Dataset):
    def __init__(self, rows, vocabulary, normalizer=None):
        self.rows = rows
        self.vocabulary = vocabulary
        self.normalizer = normalizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        values = np.load(row["_path"], allow_pickle=False)
        try:
            validate_feature_array(values)
        except ValueError as exc:
            raise ValueError(f"invalid feature array for clip {row['clip_id']}: {exc}") from exc
        if self.normalizer is not None:
            values = self.normalizer.transform(values)
        gloss_tokens = row["_gloss"].split()
        if not gloss_tokens:
            raise ValueError(f"empty token sequence for clip {row['clip_id']}")
        # Held-out signers may contain genuinely unseen glosses. Map those
        # labels to an explicit <unk> class; never add held-out labels to vocab.
        minimum_frames = len(gloss_tokens) + sum(
            left == right for left, right in zip(gloss_tokens, gloss_tokens[1:])
        )
        if minimum_frames > values.shape[0]:
            raise ValueError(f"clip {row['clip_id']} has too few frames for a valid CTC alignment")
        tokens = [self.vocabulary.get(token, self.vocabulary["<unk>"]) for token in gloss_tokens]
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



def run_epoch(model, loader, loss_fn, optimizer, device):
    """Run one training epoch and return clip-weighted mean CTC loss."""
    model.train()
    total_loss = 0.0
    count = 0
    for features, lengths, targets, target_lengths in loader:
        features = features.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad(set_to_none=True)
        log_probs, output_lengths = model(features, lengths)
        loss = loss_fn(log_probs, targets, output_lengths, target_lengths)
        if not torch.isfinite(loss):
            raise ValueError("non-finite CTC training loss; check features and gloss alignments")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = int(features.shape[0])
        total_loss += float(loss.detach()) * batch_size
        count += batch_size
    if count == 0:
        raise ValueError("training loader is empty")
    return total_loss / count


def evaluate(model, loader, loss_fn, device, reverse_vocabulary):
    model.eval()
    total_loss = 0.0
    count = 0
    word_errors = word_total = char_errors = char_total = 0
    in_vocab_word_errors = in_vocab_word_total = 0
    in_vocab_char_errors = in_vocab_char_total = 0
    oov_reference_tokens = oov_reference_clips = 0
    predicted_frame_classes = Counter()
    decoded_gloss_classes = Counter()
    predicted_frame_count = decoded_gloss_count = nonempty_hypothesis_clips = 0
    with torch.no_grad():
        for features, lengths, targets, target_lengths in loader:
            features, lengths = features.to(device), lengths.to(device)
            targets, target_lengths = targets.to(device), target_lengths.to(device)
            log_probs, output_lengths = model(features, lengths)
            if not torch.isfinite(log_probs).all():
                raise ValueError("non-finite model output during evaluation; check the checkpoint and features")
            loss = loss_fn(log_probs, targets, output_lengths, target_lengths)
            if not torch.isfinite(loss):
                raise ValueError("non-finite CTC evaluation loss; check features and gloss alignments")
            total_loss += float(loss) * features.shape[0]
            count += features.shape[0]
            predictions = log_probs.argmax(dim=-1).transpose(0, 1)
            offset = 0
            for index, target_length in enumerate(target_lengths.tolist()):
                reference_ids = targets[offset:offset + target_length].tolist()
                offset += target_length
                frame_ids = predictions[index, :int(output_lengths[index])].tolist()
                predicted_frame_classes.update(frame_ids)
                predicted_frame_count += len(frame_ids)
                predicted_ids = greedy_ctc_decode(frame_ids)
                decoded_gloss_classes.update(predicted_ids)
                decoded_gloss_count += len(predicted_ids)
                nonempty_hypothesis_clips += int(bool(predicted_ids))
                reference = [reverse_vocabulary[token] for token in reference_ids]
                hypothesis = [reverse_vocabulary[token] for token in predicted_ids]
                word_errors += edit_distance(reference, hypothesis)
                word_total += len(reference)
                ref_text, hyp_text = " ".join(reference), " ".join(hypothesis)
                char_errors += edit_distance(list(ref_text), list(hyp_text))
                char_total += len(ref_text)
                oov_count = reference.count("<unk>")
                if oov_count:
                    oov_reference_tokens += oov_count
                    oov_reference_clips += 1
                else:
                    in_vocab_word_errors += edit_distance(reference, hypothesis)
                    in_vocab_word_total += len(reference)
                    in_vocab_char_errors += edit_distance(list(ref_text), list(hyp_text))
                    in_vocab_char_total += len(ref_text)
    if count == 0:
        raise ValueError("evaluation loader is empty")
    return {
        "ctc_loss": total_loss / count,
        "gloss_wer": word_errors / max(word_total, 1),
        "gloss_cer": char_errors / max(char_total, 1),
        "clips": count,
        "oov_reference_tokens": oov_reference_tokens,
        "oov_reference_clips": oov_reference_clips,
        "in_vocabulary_only_clips": count - oov_reference_clips,
        "in_vocabulary_only_gloss_wer": in_vocab_word_errors / max(in_vocab_word_total, 1),
        "in_vocabulary_only_gloss_cer": in_vocab_char_errors / max(in_vocab_char_total, 1),
        "predicted_frame_count": predicted_frame_count,
        "blank_frame_fraction": predicted_frame_classes.get(0, 0) / max(predicted_frame_count, 1),
        "decoded_gloss_token_count": decoded_gloss_count,
        "nonempty_hypothesis_clips": nonempty_hypothesis_clips,
        "top_predicted_frame_classes": [
            {"token": reverse_vocabulary.get(token, str(token)), "frames": frames}
            for token, frames in predicted_frame_classes.most_common(12)
        ],
        "top_decoded_gloss_tokens": [
            {"token": reverse_vocabulary.get(token, str(token)), "count": occurrences}
            for token, occurrences in decoded_gloss_classes.most_common(12)
        ],
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
    parser.add_argument(
        "--causal", action="store_true",
        help="train with causal attention (still full-window inference, not a streaming runtime)",
    )
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
    vocabulary = {"<blank>": 0, "<unk>": 1, **{token: i + 2 for i, token in enumerate(vocabulary_tokens)}}
    feature_dim = None
    # Preflight every split before training. This catches held-out gloss tokens,
    # corrupt arrays and shape mismatches before a DataLoader fails mid-run.
    for row in rows:
        array = np.load(row["_path"], mmap_mode="r", allow_pickle=False)
        try:
            _, array_feature_dim = validate_feature_array(array)
        except ValueError as exc:
            raise ValueError(f"invalid feature array for clip {row['clip_id']}: {exc}") from exc
        feature_dim = feature_dim or array_feature_dim
        if int(array.shape[1]) != feature_dim:
            raise ValueError(f"feature dimensions differ across splits (clip {row['clip_id']})")
        tokens = row["_gloss"].split()
        # Keep the signer-disjoint split intact. Unseen validation/test labels
        # are represented by <unk> and counted in the report below.
        minimum_frames = len(tokens) + sum(
            left == right for left, right in zip(tokens, tokens[1:])
        )
        if minimum_frames > array.shape[0]:
            raise ValueError(f"clip {row['clip_id']} has too few frames for a valid CTC alignment")
    if feature_dim is None:
        raise ValueError("no feature arrays found")

    # Fit normalization on training signers only; validation/test never affect stats.
    normalizer = LandmarkNormalizer.fit(train_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LandmarkCTCModel(feature_dim, len(vocabulary), causal=args.causal).to(device)
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    train_loader = DataLoader(LandmarkDataset(train_rows, vocabulary, normalizer), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(LandmarkDataset(val_rows, vocabulary, normalizer), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(LandmarkDataset(test_rows, vocabulary, normalizer), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

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
            checkpoint = {
                "state_dict": model.state_dict(),
                "feature_dim": feature_dim,
                "vocabulary": vocabulary,
                "sign_language": args.language,
                "model_type": "landmark_ctc_transformer_baseline",
                "architecture": {"width": 256, "heads": 4, "layers": 4, "dropout": 0.1, "causal": args.causal},
                # Required by every future inference adapter: use these exact
                # train-split statistics rather than refitting on live frames.
                "normalization": {
                    "method": "coordinate_zscore_observed_landmarks",
                    "mean": normalizer.mean.tolist(),
                    "scale": normalizer.scale.tolist(),
                    "missing_landmark": "preserve_zero_triple",
                },
                "seed": args.seed,
            }
            # Write beside the destination, then atomically replace it. An
            # interrupted epoch must not leave a truncated "best" checkpoint.
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=args.output.name + ".", suffix=".tmp", dir=args.output.parent
            )
            os.close(descriptor)
            try:
                torch.save(checkpoint, temporary_name)
                os.replace(temporary_name, args.output)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
    checkpoint = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate(model, test_loader, loss_fn, device, reverse_vocabulary)
    oov_by_split = {}
    for split_name, split_rows in (("train", train_rows), ("validation", val_rows), ("test", test_rows)):
        unseen = sorted({token for row in split_rows for token in row["_gloss"].split()} - set(vocabulary_tokens))
        oov_by_split[split_name] = {"unique_unseen_glosses": unseen, "unique_unseen_count": len(unseen)}
    report = {
        "language": args.language,
        "oov_by_split": oov_by_split,
        "device": str(device),
        "normalization": {
            "method": "coordinate_zscore_observed_landmarks",
            "mean": normalizer.mean.tolist(),
            "scale": normalizer.scale.tolist(),
            "missing_landmark": "preserve_zero_triple",
            "fit_split": "train_only",
        },
        "test_metrics": test_metrics,
        "history": history,
        "warning": "Gloss WER/CER measure this experimental gloss baseline only, not sentence translation quality or real-world sign comprehension.",
    }
    args.output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"test_metrics": test_metrics, "checkpoint": str(args.output)}))


if __name__ == "__main__":
    main()
