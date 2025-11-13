"""Utility functions for sequence encoding and processing."""

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def encode_sequence(sequence: str, sequence_length: int = 4096) -> torch.Tensor:
    """Encode DNA sequence to one-hot tensor.

    Args:
        sequence: DNA sequence string.
        sequence_length: Target sequence length (will pad or truncate).

    Returns:
        One-hot encoded sequence with shape (4, sequence_length).
    Raises:
        ValueError: If the sequence contains unknown bases.
    """
    base_to_index = {"A": 0, "T": 1, "G": 2, "C": 3}

    # Normalise and truncate
    sequence = sequence.upper()[:sequence_length]

    # Check for unknown bases:
    unknown_bases = set(sequence) - set(base_to_index.keys())
    if unknown_bases:
        raise ValueError(f"Unknown base(s) found in sequence: {', '.join(sorted(unknown_bases))}")

    encoded = torch.zeros(4, sequence_length, dtype=torch.float32)

    for i, base in enumerate(sequence):
        idx = base_to_index.get(base)
        encoded[idx, i] = 1.0

    return encoded

