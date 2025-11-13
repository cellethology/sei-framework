"""Factories for creating fake data."""

import random


def make_test_sequence(length: int = 400) -> str:
    """Return a random DNA sequence of given length (composed of A, T, G, C)."""
    return "".join(random.choices("ATGC", k=length))


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
