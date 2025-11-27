# SEI Framework

SEI (Sequence-to-Effect Inference) is a deep learning framework for predicting chromatin profiles and sequence classes from DNA sequences. This framework allows you to extract embeddings from DNA sequences using the pre-trained SEI model.

## Overview

The SEI framework provides tools to:
- Extract embeddings from FASTA sequences using the pre-trained SEI model
- Predict chromatin profiles and sequence classes from DNA sequences
- Process sequences in batches for efficient inference

## Prerequisites

- Python 3.9 or higher
- `uv` package manager (install from [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv))
- CUDA-capable GPU (optional, but recommended for faster inference)

## Environment Setup

### 1. Install uv (if not already installed)

```bash
# Or using pip
pip install uv
```

### 2. Create Virtual Environment and Install Dependencies

Run the environment setup script:

```bash
uv sync
```

This will:
- Create a Python 3.9 virtual environment using `uv`
- Install PyTorch with CUDA 12.1 support
- Install required dependencies (selene, selene_sdk, docopt, setuptools, safetensors, tqdm)
- Install project dependencies from `pyproject.toml`

### 3. Download the SEI Model

The trained SEI model needs to be downloaded from Zenodo. You can use the provided download script:

```bash
bash download_data.sh
```

This will download:
- The trained SEI model (`sei_model.tar.gz`) from [Zenodo](https://zenodo.org/record/4906997)
- SEI framework resources (FASTA files) from [Zenodo](https://zenodo.org/record/4906962)

**Note:** The model files should be extracted to the `model/` directory. The main model file should be at `model/sei.pth`.

Alternatively, you can manually download from:
- Model: https://zenodo.org/record/4906996 (DOI: 10.5281/zenodo.4906996)
- Extract the model files to the `model/` directory

## Usage

### Extract Embeddings from FASTA Sequences

The main script for extracting embeddings is `retrieve_embeddings/retrieve_embeddings.py`.

#### Basic Usage

```bash
python retrieve_embeddings/retrieve_embeddings.py \
    --input-file retrieve_embeddings/test.fasta \
    --output-file output/embeddings.npz
```

#### Full Command with All Options

```bash
python retrieve_embeddings/retrieve_embeddings.py \
    --input-file <path-to-input.fasta> \
    --output-file <path-to-output.npz> \
    --model-path model/sei.pth \
    --batch-size 32 \
    --sequence-length 4096 \
    --use-hooks
```

#### Command-Line Arguments

- `--input-file` (required): Path to input FASTA file containing DNA sequences
- `--output-file` (required): Path to output `.npz` file where embeddings will be saved
- `--model-path` (optional): Path to SEI model file (default: `model/sei.pth`)
- `--batch-size` (optional): Batch size for processing sequences (default: 32)
- `--sequence-length` (optional): Target sequence length for encoding (default: 4096)
- `--use-hooks` (default): Use register_hooks method for embedding extraction (recommended)
- `--no-use-hooks`: Use manual layer-by-layer method instead of hooks

#### Output Format

The script outputs a compressed NumPy archive (`.npz`) file containing:
- `ids`: Array of sequence IDs from the FASTA file
- `embeddings`: Array of embeddings with shape `(num_sequences, 960, 16)`

#### Example

```bash
# Extract embeddings using the default method (hooks)
python retrieve_embeddings/retrieve_embeddings.py \
    --input-file retrieve_embeddings/test.fasta \
    --output-file output/embeddings.npz

# Extract embeddings using manual method
python retrieve_embeddings/retrieve_embeddings.py \
    --input-file retrieve_embeddings/test.fasta \
    --output-file output/embeddings.npz \
    --no-use-hooks
```

### Loading Embeddings

You can load the saved embeddings in Python:

```python
import numpy as np

# Load embeddings
data = np.load('output/embeddings.npz')
sequence_ids = data['ids']
embeddings = data['embeddings']

print(f"Loaded {len(sequence_ids)} sequences")
print(f"Embeddings shape: {embeddings.shape}")  # (num_sequences, 960, 16)
```

## Project Structure

```
sei-framework/
├── model/                # SEI model files
│   ├── sei.py            # Model architecture
│   ├── sei.pth           # Trained model weights (download required)
│   └── *.names           # Target and sequence class names
├── retrieve_embeddings/  # Embedding extraction scripts
│   ├── retrieve_embeddings.py  # Main embedding extraction script
│   ├── util.py           # Utility functions for embeddings
│   └── test.fasta        # Example input file
├── encode/               # Sequence encoding utilities
├── pca/                  # PCA analysis tools
├── tests/                # Unit tests
├── env_setup.sh          # Environment setup script
├── download_data.sh      # Model download script
└── pyproject.toml        # Project dependencies
```

## Testing

Run the test suite to verify your installation:

```bash
pytest tests/
```

## Troubleshooting

### Model File Not Found

If you encounter `FileNotFoundError: Model file not found: model/sei.pth`, make sure you have:
1. Downloaded the model using `bash download_data.sh`
2. Extracted the model files to the `model/` directory
3. Verified that `model/sei.pth` exists

The framework will automatically use CPU if CUDA is not available.

### Memory Issues

If you encounter out-of-memory errors:
- Reduce the `--batch-size` parameter (e.g., `--batch-size 16` or `--batch-size 8`)
- Process sequences in smaller chunks
