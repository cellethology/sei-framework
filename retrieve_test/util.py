"""Utility functions for extracting embeddings from Sei model.

This module provides two methods for extracting embeddings:
1. extract_embeddings_with_hooks: Uses register_hooks (clean approach)
2. extract_embeddings_manual: Manually passes through each layer (clumsy way)
"""

import inspect
import os
import sys

import torch
from torch import nn

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


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
        module = named_modules[name] if name in named_modules else getattr(model, name)

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
        sig = inspect.signature(model.forward)

        # Preferred path: model natively returns embeddings
        if "return_embeddings" in sig.parameters:
            embeddings = model(sequences, return_embeddings=True)
            return embeddings.view(embeddings.size(0), 960, -1)

        # Fallback path: hook into spline_tr to get final embedding-like tensor
        activations, handles = register_keys(model, ["spline_tr"])
        try:
            _ = model(sequences)
            spline_out = activations["spline_tr"]
            embeddings = spline_out.view(spline_out.size(0), 960, -1)
        finally:
            # Clean up hooks
            for handle in handles:
                handle.remove()

    return embeddings


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
        sig = inspect.signature(model.forward)

        # Preferred path: model natively returns embeddings
        if "return_embeddings" in sig.parameters:
            embeddings = model(sequences, return_embeddings=True)
            return embeddings.view(embeddings.size(0), 960, -1)

        # Manual approach: pass through each layer step by step
        # This replicates the forward pass from model/sei.py
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
        spline_out = model.spline_tr(out)
        return spline_out.view(spline_out.size(0), 960, model._spline_df)
