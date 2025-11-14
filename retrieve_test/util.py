"""Utility functions for extracting embeddings from Sei model.

This module provides two methods for extracting embeddings:
1. extract_embeddings_with_hooks: Uses register_hooks (clean approach)
2. extract_embeddings_manual: Manually passes through each layer (clumsy way)
"""

import inspect
import os
import sys
from pathlib import Path
from typing import Union

import pytest
import torch
from torch import nn

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def _try_native_embeddings(
    model: nn.Module, sequences: torch.Tensor
) -> Union[torch.Tensor, None]:
    """Try to extract embeddings using native return_embeddings parameter.

    Args:
        model: Loaded Sei model in eval mode.
        sequences: Input sequences with shape (batch_size, 4, sequence_length).

    Returns:
        Embeddings tensor if model supports return_embeddings, None otherwise.
    """
    sig = inspect.signature(model.forward)
    if "return_embeddings" in sig.parameters:
        embeddings = model(sequences, return_embeddings=True)
        return embeddings.view(embeddings.size(0), 960, -1)
    return None


def _manual_forward_pass(model: nn.Module, sequences: torch.Tensor) -> torch.Tensor:
    """Manually pass sequences through model layers to get spline_tr output.

    This replicates the forward pass from model/sei.py step-by-step.

    Args:
        model: Loaded Sei model in eval mode.
        sequences: Input sequences with shape (batch_size, 4, sequence_length).

    Returns:
        Output from spline_tr layer.
    """
    lout1 = model.lconv1(sequences)
    out1 = model.conv1(lout1)
    lout2 = model.lconv2(out1 + lout1)
    out2 = model.conv2(lout2)
    lout3 = model.lconv3(out2 + lout2)
    out3 = model.conv3(lout3)
    dconv_out1 = model.dconv1(out3 + lout3)
    cat_out1 = out3 + dconv_out1
    dconv_out2 = model.dconv2(cat_out1)
    cat_out2 = cat_out1 + dconv_out2
    dconv_out3 = model.dconv3(cat_out2)
    cat_out3 = cat_out2 + dconv_out3
    dconv_out4 = model.dconv4(cat_out3)
    cat_out4 = cat_out3 + dconv_out4
    dconv_out5 = model.dconv5(cat_out4)
    out = cat_out4 + dconv_out5
    return model.spline_tr(out)


def register_keys(
    model: nn.Module,
    layer_names: list[str],
) -> tuple[dict[str, torch.Tensor], list[torch.utils.hooks.RemovableHandle]]:
    """Register forward hooks on given layer names and collect their outputs.

    Args:
        model: PyTorch model.
        layer_names: Names of submodules to hook. Can be keys from
            model.named_modules() (e.g. "spline_tr" or "block3.spline_tr")
            or direct attributes.

    Returns:
        activations: Mapping layer name -> output tensor.
        handles: Hook handles (must be removed after forward).
    """
    activations: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    named_modules = dict(model.named_modules())

    for name in layer_names:
        module = named_modules.get(name) or getattr(model, name)

        def hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            key: str = name,
        ) -> None:
            activations[key] = output

        handle = module.register_forward_hook(hook)
        handles.append(handle)

    return activations, handles


def extract_embeddings_with_hooks(
    model: nn.Module,
    sequences: torch.Tensor,
) -> torch.Tensor:
    """Extract embeddings using register_hooks (clean approach).

    This method uses forward hooks to capture activations from the spline_tr layer
    without modifying the model's forward pass.

    Args:
        model: Loaded Sei model in eval mode.
        sequences: Input sequences with shape (batch_size, 4, sequence_length).

    Returns:
        Embeddings with shape (batch_size, 960, 16).
    """
    with torch.no_grad():
        # Try native embeddings first
        embeddings = _try_native_embeddings(model, sequences)
        if embeddings is not None:
            return embeddings

        # Fallback: hook into spline_tr to get final embedding-like tensor
        activations, handles = register_keys(model, ["spline_tr"])
        try:
            _ = model(sequences)
            spline_out = activations["spline_tr"]
            return spline_out.view(spline_out.size(0), 960, -1)
        finally:
            for handle in handles:
                handle.remove()


def extract_embeddings_manual(
    model: nn.Module,
    sequences: torch.Tensor,
) -> torch.Tensor:
    """Extract embeddings by manually passing through each layer (clumsy way).

    This method manually calls each layer of the model to extract embeddings.
    This is the "clumsy" approach that replicates the forward pass step-by-step.

    Args:
        model: Loaded Sei model in eval mode.
        sequences: Input sequences with shape (batch_size, 4, sequence_length).

    Returns:
        Embeddings with shape (batch_size, 960, 16).
    """
    with torch.no_grad():
        # Try native embeddings first
        embeddings = _try_native_embeddings(model, sequences)
        if embeddings is not None:
            return embeddings

        # Manual approach: pass through each layer step by step
        spline_out = _manual_forward_pass(model, sequences)
        return spline_out.view(spline_out.size(0), 960, model._spline_df)


def inference_sequences(
    model: nn.Module,
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
    from embeddings.util import encode_sequence

    extract_fn = extract_embeddings_with_hooks if use_hooks else extract_embeddings_manual
    batch_size = batch_size or 1  # Normalize None to 1 for unified processing

    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        encoded_list = [encode_sequence(seq, sequence_length=sequence_length) for seq in batch_seqs]
        batch_tensor = torch.stack(encoded_list)  # (batch_size, 4, sequence_length)
        batch_embeddings = extract_fn(model, batch_tensor)
        all_embeddings.append(batch_embeddings)

    return torch.cat(all_embeddings, dim=0)  # (num_sequences, 960, 16)


def load_model(model_path: str) -> nn.Module:
    """Load and initialize the Sei model.

    Args:
        model_path: Path to the trained SEI model (.pth file).

    Returns:
        Loaded and initialized Sei model in eval mode.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
    """
    from model.sei import Sei

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = Sei()
    state_dict = torch.load(model_path, map_location="cpu")

    # Remove 'module.model.' prefix from keys if present
    prefix = "module.model."
    new_state_dict = {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }

    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def load_test_model(model_path: str = "model/sei.pth") -> nn.Module:
    """Load Sei model for testing.

    Args:
        model_path: Path to the model checkpoint file. Defaults to "model/sei.pth".

    Returns:
        Loaded Sei model in eval mode.

    Raises:
        pytest.skip: If the model file doesn't exist.
    """
    try:
        return load_model(model_path)
    except FileNotFoundError:
        pytest.skip(f"Model file not found: {model_path}")
