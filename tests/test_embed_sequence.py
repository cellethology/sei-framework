"""Tests to verify sequence order is preserved when extracting embeddings.

This module tests that:
1. Processing sequences one-by-one produces the same results as batch processing
2. The order of embeddings matches the order of input sequences
3. Both extraction methods (hooks and manual) produce consistent results
"""

import torch

from retrieve_test.retrieve_embeddings import retrieve_embeddings_from_sequences  # noqa: E402
from retrieve_test.util import load_test_model


def test_one_by_one_vs_batch_with_hooks(test_sequences):
    """Test that one-by-one inference matches batch inference using hooks method."""
    model = load_test_model()

    # Process sequences one by one
    one_by_one_embeddings = retrieve_embeddings_from_sequences(
        model, test_sequences, sequence_length=400, batch_size=1, use_hooks=True
    )

    # Process sequences in a batch
    batch_embeddings = retrieve_embeddings_from_sequences(
        model, test_sequences, sequence_length=400, batch_size=len(test_sequences), use_hooks=True
    )

    # Verify order and equality
    assert one_by_one_embeddings.shape[0] == len(test_sequences)
    assert batch_embeddings.shape[0] == len(test_sequences)

    # Compare each position: one-by-one vs batch
    for i in range(len(test_sequences)):
        assert torch.allclose(
            one_by_one_embeddings[i], batch_embeddings[i], atol=1e-5
        ), f"Embedding {i} from one-by-one processing doesn't match batch processing"

    # Verify embeddings are different (they should be for different sequences)
    assert not torch.allclose(
        batch_embeddings[0], batch_embeddings[1], atol=1e-5
    ), "Embeddings for different sequences should be different"


def test_one_by_one_vs_batch_manual(test_sequences):
    """Test that one-by-one inference matches batch inference using manual method."""
    model = load_test_model()

    # Process sequences one by one
    one_by_one_embeddings = retrieve_embeddings_from_sequences(
        model, test_sequences, sequence_length=400, batch_size=1, use_hooks=False
    )

    # Process sequences in a batch
    batch_embeddings = retrieve_embeddings_from_sequences(
        model, test_sequences, sequence_length=400, batch_size=len(test_sequences), use_hooks=False
    )

    # Verify order and equality
    assert one_by_one_embeddings.shape[0] == len(test_sequences)
    assert batch_embeddings.shape[0] == len(test_sequences)

    # Compare each position: one-by-one vs batch
    for i in range(len(test_sequences)):
        assert torch.allclose(
            one_by_one_embeddings[i], batch_embeddings[i], atol=1e-5
        ), f"Embedding {i} from one-by-one processing doesn't match batch processing"

    # Verify embeddings are different (they should be for different sequences)
    assert not torch.allclose(
        batch_embeddings[0], batch_embeddings[1], atol=1e-5
    ), "Embeddings for different sequences should be different"


def test_hooks_vs_manual_produce_same_results(test_sequences):
    """Test that hooks method and manual method produce identical results."""
    model = load_test_model()

    # Extract using both methods
    hooks_embeddings = retrieve_embeddings_from_sequences(
        model, test_sequences, sequence_length=400, batch_size=len(test_sequences), use_hooks=True
    )
    manual_embeddings = retrieve_embeddings_from_sequences(
        model, test_sequences, sequence_length=400, batch_size=len(test_sequences), use_hooks=False
    )

    # Verify they produce the same results
    assert hooks_embeddings.shape == manual_embeddings.shape
    assert torch.allclose(
        hooks_embeddings, manual_embeddings, atol=1e-5
    ), "Hooks and manual methods should produce identical embeddings"


def test_batch_order_preservation(test_sequences):
    """Test that batch processing preserves the exact order of input sequences."""
    model = load_test_model()

    # Process one by one
    one_by_one_embeddings = retrieve_embeddings_from_sequences(
        model, test_sequences, sequence_length=400, batch_size=1, use_hooks=True
    )

    # Process in batch
    batch_embeddings = retrieve_embeddings_from_sequences(
        model, test_sequences, sequence_length=400, batch_size=len(test_sequences), use_hooks=True
    )

    # Verify order: batch embeddings should match one-by-one in the same order
    for i in range(len(test_sequences)):
        assert torch.allclose(
            one_by_one_embeddings[i], batch_embeddings[i], atol=1e-5
        ), f"Order mismatch: sequence {i} doesn't match batch result"

    # Verify we can identify sequences by their embeddings
    # Each sequence should have a unique embedding
    for i in range(len(test_sequences)):
        for j in range(i + 1, len(test_sequences)):
            assert not torch.allclose(
                batch_embeddings[i], batch_embeddings[j], atol=1e-5
            ), f"Sequences {i} and {j} should have different embeddings"


def test_batch_processing_with_different_batch_sizes(test_sequences):
    """Test that batch processing works correctly with different batch sizes."""
    model = load_test_model()

    # Process all sequences one by one
    one_by_one_embeddings = retrieve_embeddings_from_sequences(
        model, test_sequences, sequence_length=400, batch_size=1, use_hooks=True
    )

    # Process in batches of different sizes
    batch_size_2_embeddings = retrieve_embeddings_from_sequences(
        model, test_sequences, sequence_length=400, batch_size=2, use_hooks=True
    )
    batch_size_3_embeddings = retrieve_embeddings_from_sequences(
        model, test_sequences, sequence_length=400, batch_size=3, use_hooks=True
    )

    # All methods should produce the same results in the same order
    assert (
        one_by_one_embeddings.shape[0]
        == batch_size_2_embeddings.shape[0]
        == batch_size_3_embeddings.shape[0]
    )

    for i in range(len(test_sequences)):
        assert torch.allclose(
            one_by_one_embeddings[i], batch_size_2_embeddings[i], atol=1e-5
        ), f"One-by-one vs batch_size=2 mismatch at position {i}"
        assert torch.allclose(
            one_by_one_embeddings[i], batch_size_3_embeddings[i], atol=1e-5
        ), f"One-by-one vs batch_size=3 mismatch at position {i}"


def test_sequence_order_with_register_hooks(test_sequences):
    """Test that register_hooks preserves sequence order in batch processing."""
    # Use first 3 sequences from the fixture
    sequences = test_sequences[:3]

    model = load_test_model()

    # Process one by one with hooks
    one_by_one_embeddings = retrieve_embeddings_from_sequences(
        model, sequences, sequence_length=400, batch_size=1, use_hooks=True
    )

    # Process in batch with hooks
    batch_embeddings = retrieve_embeddings_from_sequences(
        model, sequences, sequence_length=400, batch_size=len(sequences), use_hooks=True
    )

    # Verify order is preserved
    assert one_by_one_embeddings.shape[0] == batch_embeddings.shape[0]

    for i in range(len(sequences)):
        assert torch.allclose(
            one_by_one_embeddings[i], batch_embeddings[i], atol=1e-5
        ), f"Order mismatch at position {i}: one-by-one doesn't match batch"


def test_manual_method_preserves_order(test_sequences):
    """Test that manual layer-by-layer method preserves sequence order."""
    model = load_test_model()

    # Process one by one
    one_by_one_embeddings = retrieve_embeddings_from_sequences(
        model, test_sequences, sequence_length=400, batch_size=1, use_hooks=False
    )

    # Process in batch
    batch_embeddings = retrieve_embeddings_from_sequences(
        model, test_sequences, sequence_length=400, batch_size=len(test_sequences), use_hooks=False
    )

    # Verify order and equality
    for i in range(len(test_sequences)):
        assert torch.allclose(
            one_by_one_embeddings[i], batch_embeddings[i], atol=1e-5
        ), f"Manual method: order mismatch at position {i}"
