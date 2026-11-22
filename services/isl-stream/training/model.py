"""Small CTC temporal baseline for landmark-sequence experiments.

Input is pre-extracted, signer-reviewed feature arrays, not raw RGB video.
This is a trainable research baseline, not a pretrained translator.
"""
from __future__ import annotations

import math
import torch
from torch import nn


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, width: int, max_length: int = 4096):
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scales = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32)
            * (-math.log(10000.0) / width)
        )
        encoding = torch.zeros(max_length, width)
        encoding[:, 0::2] = torch.sin(positions * scales)
        if width > 1:
            encoding[:, 1::2] = torch.cos(positions * scales[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] > self.encoding.shape[1]:
            raise ValueError("sequence exceeds configured positional encoding length")
        return values + self.encoding[:, : values.shape[1]].to(values.dtype)


class LandmarkCTCModel(nn.Module):
    """Transformer encoder producing framewise gloss logits for CTC decoding."""

    def __init__(
        self,
        feature_dim: int,
        vocabulary_size: int,
        width: int = 256,
        heads: int = 4,
        layers: int = 4,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        if feature_dim < 1 or vocabulary_size < 2:
            raise ValueError("feature_dim and vocabulary_size must be positive")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if not isinstance(causal, bool):
            raise ValueError("causal must be a boolean")
        self.causal = causal
        self.input_projection = nn.Linear(feature_dim, width)
        self.position = SinusoidalPositionEncoding(width)
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, vocabulary_size)

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or lengths.ndim != 1:
            raise ValueError("features must be [batch,time,features] and lengths [batch]")
        if features.shape[0] != lengths.shape[0]:
            raise ValueError("batch dimensions do not match")
        if torch.any(lengths <= 0) or torch.any(lengths > features.shape[1]):
            raise ValueError("sequence lengths are outside padded tensor bounds")
        positions = torch.arange(features.shape[1], device=features.device)
        padding_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)
        hidden = self.position(self.input_projection(features))
        attention_mask = None
        if self.causal:
            attention_mask = torch.triu(
                torch.ones(features.shape[1], features.shape[1], dtype=torch.bool, device=features.device),
                diagonal=1,
            )
        hidden = self.encoder(
            hidden, mask=attention_mask, src_key_padding_mask=padding_mask
        )
        logits = self.classifier(self.norm(hidden)).log_softmax(dim=-1)
        return logits.transpose(0, 1), lengths
