"""Pytest configuration and shared fixtures."""

import pytest

from tests.factories import (
    make_sequences_distinct_patterns,
    make_sequences_mixed_patterns,
    make_sequences_variable,
)


@pytest.fixture
def test_sequences() -> list[str]:
    """Fixture providing test sequences with distinct patterns."""
    return make_sequences_distinct_patterns(repeats=100)


@pytest.fixture
def test_sequences_mixed() -> list[str]:
    """Fixture providing test sequences with mixed patterns."""
    return make_sequences_mixed_patterns(repeats=100)


@pytest.fixture
def test_sequences_short() -> list[str]:
    """Fixture providing shorter test sequences."""
    return make_sequences_distinct_patterns(repeats=50)


@pytest.fixture
def test_sequences_variable() -> list[str]:
    """Fixture providing variable test sequences."""
    return make_sequences_variable(count=6, repeats=50)
