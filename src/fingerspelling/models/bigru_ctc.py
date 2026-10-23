"""Notebook-compatible BiGRU-CTC model used by the FSboard baselines."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiGRUCTC(nn.Module):
    """Bidirectional GRU followed by a per-frame CTC classifier.

    Inputs use ``[batch, time, features]`` and outputs are raw logits with
    shape ``[time, batch, classes]``, matching the notebook's CTC convention.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 60,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_size <= 0 or num_layers <= 0 or num_classes <= 1:
            raise ValueError("Model dimensions must be positive; num_classes must exceed one")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        recurrent_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(recurrent_dim, num_classes)

    def forward(self, features: Tensor, lengths: Tensor | None = None) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, features]")
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} input features, got {features.shape[-1]}"
            )
        batch_size, time_steps, _ = features.shape
        if lengths is None:
            lengths = torch.full((batch_size,), time_steps, dtype=torch.long)
        lengths = torch.as_tensor(lengths, dtype=torch.long, device="cpu")
        if lengths.shape != (batch_size,):
            raise ValueError("lengths must contain one sequence length per batch item")
        if bool((lengths <= 0).any()) or bool((lengths > time_steps).any()):
            raise ValueError("each sequence length must be in [1, time]")

        packed = pack_padded_sequence(
            features,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed)
        recurrent, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=time_steps,
        )
        logits = self.classifier(self.dropout(recurrent))
        return logits.transpose(0, 1).contiguous()

