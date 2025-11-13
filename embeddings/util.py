"""Utility functions for sequence encoding and processing."""

import torch

# Base-to-index mapping: A -> 0, C -> 1, G -> 2, T -> 3
# Note: This differs from Selene SDK's original mapping (A=0, T=1, G=2, C=3)
# Reference: https://selene.flatironinstitute.org/master/sequences.html#sequence-to-encoding
BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode_sequence(sequence: str, sequence_length: int = 4096) -> torch.Tensor:
    """Encode DNA sequence to one-hot tensor.

    This function encodes DNA sequences to one-hot tensors. The base-to-index
    mapping is A=0, C=1, G=2, T=3.

    Note: Selene SDK's sequence_to_encoding() outputs shape (sequence_length, 4),
    but internally transposes it to (4, sequence_length) before passing to PyTorch
    models. This function directly outputs (4, sequence_length) which is the format
    expected by PyTorch Conv1d layers.

    Args:
        sequence: DNA sequence string.
        sequence_length: Target sequence length (will pad or truncate).

    Returns:
        One-hot encoded sequence with shape (4, sequence_length).
        Channel order: [A, C, G, T] corresponding to indices [0, 1, 2, 3].
    Raises:
        ValueError: If the sequence contains unknown bases.
    """
    base_to_index = BASE_TO_INDEX

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
