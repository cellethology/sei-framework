#!/usr/bin/env python
"""
Reduce embedding dimensionality using cuML PCA.

This script loads embeddings from a .npz file (output from retrieve_embeddings.py),
flattens them, and uses cuML PCA to reduce dimensionality to 512 dimensions.

Requirements:
    - cuML: Install via conda: conda install -c rapidsai -c conda-forge cuml
    - cupy: Install via pip: pip install cupy-cuda11x (adjust CUDA version as needed)

Usage:
    python pca/pca.py \
        --input-file output/embeddings.npz \
        --output-file output/embeddings_pca512.npz \
        --n-components 512
"""

import argparse
from pathlib import Path

import cupy as cp
import numpy as np
from cuml import PCA


def load_embeddings(input_file: str) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings and IDs from .npz file.

    Args:
        input_file: Path to input .npz file.

    Returns:
        Tuple of (embeddings_array, ids_array).
        embeddings_array has shape (num_sequences, 960, 16).
    """
    print(f"Loading embeddings from {input_file}...")
    data = np.load(input_file, allow_pickle=True)
    embeddings = data["embeddings"]
    ids = data["ids"]

    print(f"Loaded {len(ids)} sequences")
    print(f"Original embeddings shape: {embeddings.shape}")

    return embeddings, ids


def flatten_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Flatten embeddings from (N, 960, 16) to (N, 15360).

    Args:
        embeddings: Embeddings array with shape (num_sequences, 960, 16).

    Returns:
        Flattened embeddings with shape (num_sequences, 15360).
    """
    if len(embeddings.shape) == 3:
        num_sequences, height, width = embeddings.shape
        flattened = embeddings.reshape(num_sequences, height * width)
        print(f"Flattened embeddings shape: {flattened.shape}")
        return flattened
    if len(embeddings.shape) == 2:
        print(f"Embeddings already flattened: {embeddings.shape}")
        return embeddings
    raise ValueError(f"Unexpected embeddings shape: {embeddings.shape}")


def reduce_dimensionality(
    embeddings: np.ndarray, n_components: int = 512
) -> tuple[np.ndarray, PCA]:
    """Reduce embedding dimensionality using cuML PCA.

    Args:
        embeddings: Flattened embeddings array with shape (num_sequences, features).
        n_components: Number of components for PCA (default: 512).

    Returns:
        Tuple of (reduced_embeddings, pca_model).
        reduced_embeddings has shape (num_sequences, n_components).
    """
    print(f"Reducing dimensionality from {embeddings.shape[1]} to {n_components}...")

    # Convert to cupy array for GPU processing
    embeddings_gpu = cp.asarray(embeddings)

    # Fit PCA and transform
    pca = PCA(n_components=n_components, random_state=42)
    reduced_embeddings_gpu = pca.fit_transform(embeddings_gpu)

    # Convert back to numpy array
    reduced_embeddings = cp.asnumpy(reduced_embeddings_gpu)

    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    total_variance = float(cp.sum(explained_variance))
    print(f"Total explained variance: {total_variance:.4f} ({total_variance*100:.2f}%)")
    print(f"Reduced embeddings shape: {reduced_embeddings.shape}")

    return reduced_embeddings, pca


def save_reduced_embeddings(
    reduced_embeddings: np.ndarray, ids: np.ndarray, output_file: str
) -> None:
    """Save reduced embeddings to .npz file.

    Args:
        reduced_embeddings: Reduced embeddings array with shape (num_sequences, n_components).
        ids: Array of sequence IDs.
        output_file: Path to output .npz file.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving reduced embeddings to {output_file}...")
    np.savez_compressed(output_file, ids=ids, embeddings=reduced_embeddings)
    print(f"Saved {len(ids)} reduced embeddings to {output_file}")


def reduce_embeddings_pca(
    input_file: str, output_file: str, n_components: int = 512
) -> None:
    """Main function to reduce embedding dimensionality using PCA.

    Args:
        input_file: Path to input .npz file containing embeddings.
        output_file: Path to output .npz file for reduced embeddings.
        n_components: Number of PCA components (default: 512).
    """
    # Load embeddings
    embeddings, ids = load_embeddings(input_file)

    # Flatten embeddings
    flattened_embeddings = flatten_embeddings(embeddings)

    # Reduce dimensionality
    reduced_embeddings, pca_model = reduce_dimensionality(
        flattened_embeddings, n_components=n_components
    )

    # Save reduced embeddings
    save_reduced_embeddings(reduced_embeddings, ids, output_file)

    print("PCA dimensionality reduction completed successfully!")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reduce embedding dimensionality using cuML PCA"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input .npz file containing embeddings",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to output .npz file for reduced embeddings",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=512,
        help="Number of PCA components (default: 512)",
    )

    args = parser.parse_args()

    reduce_embeddings_pca(
        input_file=args.input_file,
        output_file=args.output_file,
        n_components=args.n_components,
    )


if __name__ == "__main__":
    main()
