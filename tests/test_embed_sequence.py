"""Tests to verify sequence order is preserved when extracting embeddings.

This module tests that:
1. Processing sequences one-by-one produces the same results as batch processing
2. The order of embeddings matches the order of input sequences
3. Both extraction methods (hooks and manual) produce consistent results
"""

from pathlib import Path
from typing import Union

import pytest
import torch

from embeddings.util import encode_sequence


def _inference_sequences(
    model: torch.nn.Module,
    sequences: list[str],
    sequence_length: int = 400,
    batch_size: Union[int, None] = None,
    use_hooks: bool = True,
) -> torch.Tensor:
    """Inference sequences and return embeddings.

    Args:
        model: Loaded Sei model in eval mode.
        sequences: List of DNA sequence strings.
        sequence_length: Target sequence length for encoding.
        batch_size: If None, process one-by-one. If int, process in batches of that size.
        use_hooks: If True, use register_hooks method. If False, use manual method.

    Returns:
        Embeddings tensor with shape (num_sequences, 960, 16).
    """
    from retrieve_test.util import (
        extract_embeddings_manual,
        extract_embeddings_with_hooks,
    )

    extract_fn = extract_embeddings_with_hooks if use_hooks else extract_embeddings_manual

    if batch_size is None:
        # Process one by one
        embeddings_list = []
        for seq in sequences:
            encoded = encode_sequence(seq, sequence_length=sequence_length)
            single_batch = encoded.unsqueeze(0)  # (1, 4, sequence_length)
            embedding = extract_fn(model, single_batch)
            embeddings_list.append(embedding.squeeze(0))  # Remove batch dim
        return torch.stack(embeddings_list)  # (num_sequences, 960, 16)

    # Process in batches
    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        encoded_list = [encode_sequence(seq, sequence_length=sequence_length) for seq in batch_seqs]
        batch_tensor = torch.stack(encoded_list)  # (batch_size, 4, sequence_length)
        batch_embeddings = extract_fn(model, batch_tensor)
        all_embeddings.append(batch_embeddings)

    return torch.cat(all_embeddings, dim=0)  # (num_sequences, 960, 16)


def _load_test_model(model_path: str = "model/sei.pth"):
    """Load Sei model for testing."""
    from model.sei import Sei

    model = Sei()
    if not Path(model_path).exists():
        pytest.skip(f"Model file not found: {model_path}")

    state_dict = torch.load(model_path, map_location="cpu")

    # Remove 'module.model.' prefix from keys if present
    new_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module.model."):
            new_key = key[len("module.model.") :]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def test_one_by_one_vs_batch_with_hooks(test_sequences):
    """Test that one-by-one inference matches batch inference using hooks method."""
    model = _load_test_model()

    # Process sequences one by one
    one_by_one_embeddings = _inference_sequences(
        model, test_sequences, sequence_length=400, batch_size=None, use_hooks=True
    )

    # Process sequences in a batch
    batch_embeddings = _inference_sequences(
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
    model = _load_test_model()

    # Process sequences one by one
    one_by_one_embeddings = _inference_sequences(
        model, test_sequences, sequence_length=400, batch_size=None, use_hooks=False
    )

    # Process sequences in a batch
    batch_embeddings = _inference_sequences(
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
    model = _load_test_model()

    # Extract using both methods
    hooks_embeddings = _inference_sequences(
        model, test_sequences, sequence_length=400, batch_size=len(test_sequences), use_hooks=True
    )
    manual_embeddings = _inference_sequences(
        model, test_sequences, sequence_length=400, batch_size=len(test_sequences), use_hooks=False
    )

    # Verify they produce the same results
    assert hooks_embeddings.shape == manual_embeddings.shape
    assert torch.allclose(
        hooks_embeddings, manual_embeddings, atol=1e-5
    ), "Hooks and manual methods should produce identical embeddings"


def test_batch_order_preservation(test_sequences):
    """Test that batch processing preserves the exact order of input sequences."""
    model = _load_test_model()

    # Process one by one
    one_by_one_embeddings = _inference_sequences(
        model, test_sequences, sequence_length=400, batch_size=None, use_hooks=True
    )

    # Process in batch
    batch_embeddings = _inference_sequences(
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
    model = _load_test_model()

    # Process all sequences one by one
    one_by_one_embeddings = _inference_sequences(
        model, test_sequences, sequence_length=400, batch_size=None, use_hooks=True
    )

    # Process in batches of different sizes
    batch_size_2_embeddings = _inference_sequences(
        model, test_sequences, sequence_length=400, batch_size=2, use_hooks=True
    )
    batch_size_3_embeddings = _inference_sequences(
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

    model = _load_test_model()

    # Process one by one with hooks
    one_by_one_embeddings = _inference_sequences(
        model, sequences, sequence_length=400, batch_size=None, use_hooks=True
    )

    # Process in batch with hooks
    batch_embeddings = _inference_sequences(
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
    model = _load_test_model()

    # Process one by one
    one_by_one_embeddings = _inference_sequences(
        model, test_sequences, sequence_length=400, batch_size=None, use_hooks=False
    )

    # Process in batch
    batch_embeddings = _inference_sequences(
        model, test_sequences, sequence_length=400, batch_size=len(test_sequences), use_hooks=False
    )

    # Verify order and equality
    for i in range(len(test_sequences)):
        assert torch.allclose(
            one_by_one_embeddings[i], batch_embeddings[i], atol=1e-5
        ), f"Manual method: order mismatch at position {i}"
