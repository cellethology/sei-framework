"""Factories for creating fake data."""

import random

# Fixed seed for deterministic, reproducible test sequences
_RANDOM_SEED = 42


def make_test_sequence(length: int = 400) -> str:
    """Return a deterministic DNA sequence of given length (composed of A, T, G, C).

    Uses a fixed random seed to ensure reproducibility while generating sequences
    that appear random. The same length will always produce the same sequence.

    Args:
        length: The length of the sequence to generate. Defaults to 400.

    Returns:
        A DNA sequence string of the specified length.

    Examples:
        >>> seq1 = make_test_sequence(100)
        >>> seq2 = make_test_sequence(100)
        >>> seq1 == seq2  # True - deterministic
    """
    # Create a new Random instance with fixed seed to avoid affecting global state
    rng = random.Random(_RANDOM_SEED)
    return "".join(rng.choices("ATGC", k=length))


def make_sequences_distinct_patterns(repeats: int = 100) -> list[str]:
    """Return sequences with distinct patterns (all A's, T's, G's, C's)."""
    return [
        "AAAA" * repeats,  # All A's
        "TTTT" * repeats,  # All T's
        "GGGG" * repeats,  # All G's
        "CCCC" * repeats,  # All C's
    ]


def make_sequences_mixed_patterns(repeats: int = 100) -> list[str]:
    """Return sequences with mixed patterns."""
    return [
        "ACGT" * repeats,
        "TTAA" * repeats,
        "GGCC" * repeats,
        "CATG" * repeats,
    ]


def make_sequences_variable(count: int = 6, repeats: int = 50) -> list[str]:
    """Return variable test sequences."""
    return [f"ACGT{i}" * repeats for i in range(count)]
