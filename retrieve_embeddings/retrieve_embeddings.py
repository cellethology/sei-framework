#!/usr/bin/env python
"""
Retrieve embeddings from FASTA sequences using the Sei model with register hooks.

Usage:
# Use register_hooks method (default, clean approach)
python retrieve_embeddings/retrieve_embeddings.py \
    --input-file retrieve_embeddings/test.fasta \
    --output-file output/embeddings.npz

# Use manual layer-by-layer method (clumsy way)
python retrieve_embeddings/retrieve_embeddings.py \
    --input-file retrieve_embeddings/test.fasta \
    --output-file output/embeddings.npz \
    --no-use-hooks
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO  # noqa: E402

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from retrieve_embeddings.util import inference_sequences, load_model  # noqa: E402


def retrieve_embeddings_from_sequences(
    model: torch.nn.Module,
    sequences: list[str],
    batch_size: int = 32,
    sequence_length: int = 4096,
    use_hooks: bool = True,
) -> torch.Tensor:
    """Retrieve embeddings from a list of DNA sequences.

    This is the core embedding extraction function that works with in-memory sequences.
    It can be used directly in tests or by higher-level functions that handle file I/O.

    Args:
        model: Loaded Sei model in eval mode.
        sequences: List of DNA sequence strings.
        batch_size: Batch size for processing sequences.
        sequence_length: Target sequence length for encoding.
        use_hooks: If True, use register_hooks method (clean). If False, use manual layer-by-layer method (clumsy).

    Returns:
        Embeddings tensor with shape (num_sequences, 960, 16).
    """
    return inference_sequences(
        model=model,
        sequences=sequences,
        sequence_length=sequence_length,
        batch_size=batch_size,
        use_hooks=use_hooks,
    )


def load_fasta(input_file: str) -> list[tuple[str, str]]:
    """Load sequences from FASTA file.

    Args:
        input_file: Path to FASTA file.

    Returns:
        List of tuples (sequence_id, sequence_string).
    """
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {input_file}")

    records = []
    for record in SeqIO.parse(path, "fasta"):
        records.append((record.id, str(record.seq)))

    return records


def retrieve_embeddings(
    input_file: str,
    output_file: str,
    model_path: str,
    batch_size: int = 32,
    sequence_length: int = 4096,
    use_hooks: bool = True,
) -> None:
    """Retrieve embeddings from FASTA sequences.

    Args:
        input_file: Path to input FASTA file.
        output_file: Path to output .npz file.
        model_path: Path to SEI model (.pth file).
        batch_size: Batch size for processing sequences.
        sequence_length: Target sequence length for encoding.
        use_hooks: If True, use register_hooks method (clean). If False, use manual layer-by-layer method (clumsy).
    """
    print(f"Loading FASTA sequences from {input_file}...")
    sequences_data = load_fasta(input_file)
    print(f"Loaded {len(sequences_data)} sequences")

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    method_name = "register_hooks" if use_hooks else "manual layer-by-layer"
    print(f"Using {method_name} method for embedding extraction")

    # Extract sequence IDs and sequences
    sequence_ids = [seq_id for seq_id, _ in sequences_data]
    sequences = [seq for _, seq in sequences_data]

    # Extract embeddings using retrieve_embeddings_from_sequences
    print(f"Processing sequences in batches of {batch_size}...")
    final_embeddings = retrieve_embeddings_from_sequences(
        model=model,
        sequences=sequences,
        batch_size=batch_size,
        sequence_length=sequence_length,
        use_hooks=use_hooks,
    )

    print(f"Final embeddings shape: {final_embeddings.shape}")

    # Convert to numpy arrays
    embeddings_array = final_embeddings.detach().cpu().numpy()
    ids_array = np.array(sequence_ids, dtype=object)

    # Save to .npz file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving embeddings to {output_file}...")
    np.savez_compressed(output_file, ids=ids_array, embeddings=embeddings_array)

    print(f"Saved {len(sequence_ids)} embeddings to {output_file}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Retrieve embeddings from FASTA sequences using Sei model"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input FASTA file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to output .npz file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="model/sei.pth",
        help="Path to SEI model (.pth file)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing sequences",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=4096,
        help="Target sequence length for encoding",
    )
    parser.add_argument(
        "--use-hooks",
        action="store_true",
        default=True,
        help="Use register_hooks method (default: True). Use --no-use-hooks for manual method.",
    )
    parser.add_argument(
        "--no-use-hooks",
        dest="use_hooks",
        action="store_false",
        help="Use manual layer-by-layer method instead of hooks",
    )

    args = parser.parse_args()

    retrieve_embeddings(
        input_file=args.input_file,
        output_file=args.output_file,
        model_path=args.model_path,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        use_hooks=args.use_hooks,
    )


if __name__ == "__main__":
    main()
