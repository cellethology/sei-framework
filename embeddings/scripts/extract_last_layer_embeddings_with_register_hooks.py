import argparse
import inspect
import os
import sys

import pandas as pd
import torch
from safetensors.torch import save_file
from torch import nn
from tqdm import tqdm

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from embeddings.util import encode_sequence  # noqa: E402
from model.sei import Sei  # noqa: E402


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


def extract_last_embedding(
    model_path: str,
    sequences: torch.Tensor,
) -> torch.Tensor:
    """Extract the last embedding layer from SEI model.

    Args:
        model_path: Path to the trained SEI model (.pth file).
        sequences: Input sequences with shape (batch_size, 4, sequence_length).

    Returns:
        Last embedding layer features with shape (batch_size, 960, 16).
    """
    # Initialise model and load state dict
    model = Sei()
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
            for handle in handles:
                handle.remove()

    return embeddings


def process_csv_to_safetensors(
    csv_path: str,
    model_path: str,
    output_path: str,
    batch_size: int = 32,
) -> None:
    """Process CSV file and extract embeddings, save as safetensors.

    Args:
        csv_path: Path to CSV file with sequence data.
        model_path: Path to SEI model (.pth).
        output_path: Path to save safetensors file.
        batch_size: Batch size for processing.
    """
    print(f"Loading CSV data from {csv_path}...")
    df = pd.read_csv(csv_path)

    required_cols = {"full_sequence", "sequence_index", "Expression"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    print(f"Processing {len(df)} sequences...")

    all_embeddings: list[torch.Tensor] = []
    all_variant_ids: list[int] = []
    all_expressions: list[float] = []

    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_df = df.iloc[i : i + batch_size]

        # Encode sequences
        batch_sequences = [encode_sequence(seq) for seq in batch_df["full_sequence"]]
        batch_tensor = torch.stack(batch_sequences)  # (B, 4, L)

        # Extract embeddings
        embeddings = extract_last_embedding(model_path, batch_tensor)

        all_embeddings.append(embeddings)
        all_variant_ids.extend(batch_df["sequence_index"].tolist())
        all_expressions.extend(batch_df["Expression"].tolist())

    # Concatenate along batch dimension
    final_embeddings = torch.cat(all_embeddings, dim=0)

    print(f"Final embeddings shape: {final_embeddings.shape}")

    # Convert IDs to tensor (assumes they are numeric)
    variant_ids_tensor = torch.tensor(all_variant_ids, dtype=torch.long)
    expressions_tensor = torch.tensor(
        all_expressions,
        dtype=torch.float32,
    )

    save_dict = {
        "embeddings": final_embeddings,
        "variant_ids": variant_ids_tensor,
        "expressions": expressions_tensor,
    }

    print(f"Saving results to {output_path}...")
    save_file(save_dict, output_path)

    print("Processing complete!")
    print(
        f"Saved {len(all_variant_ids)} samples " f"with embeddings shape {final_embeddings.shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract embeddings from CSV sequences")
    parser.add_argument(
        "--csv_path",
        type=str,
        default=(
            "/lambda/nfs/evo1zelun/sei-framework/embeddings/"
            "input_data/all_data_with_sequence.csv"
        ),
        help="Path to CSV file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/lambda/nfs/evo1zelun/sei-framework/model/sei.pth",
        help="Path to SEI model",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=(
            "/home/ubuntu/evo1zelun/sei-framework/embeddings/output/"
            "166k_sei_embeddings.safetensors"
        ),
        help="Output path for safetensors file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing",
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    process_csv_to_safetensors(
        csv_path=args.csv_path,
        model_path=args.model_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
