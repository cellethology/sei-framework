import argparse
import os
import sys

import pandas as pd
import torch
from safetensors.torch import save_file
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.sei import Sei


# NOTE: Need to give in a different sequence length
def encode_sequence(sequence, sequence_length=4096):
    """
    Encode DNA sequence to one-hot tensor.

    Parameters
    ----------
    sequence : str
        DNA sequence string
    sequence_length : int
        Target sequence length (will pad or truncate)

    Returns
    -------
    torch.Tensor
        One-hot encoded sequence with shape (4, sequence_length)
    """
    # Mapping for DNA bases
    base_to_index = {"A": 0, "T": 1, "G": 2, "C": 3}

    # Convert to uppercase and truncate/pad
    sequence = sequence.upper()[:sequence_length]

    # Initialize one-hot tensor
    encoded = torch.zeros(4, sequence_length)

    # Fill in one-hot encoding
    for i, base in enumerate(sequence):
        if base in base_to_index:
            encoded[base_to_index[base], i] = 1.0
        # Unknown bases remain as zeros

    return encoded


def extract_last_embedding(model_path, sequences):
    """
    Extract the last embedding layer from SEI model.

    Parameters
    ----------
    model_path : str
        Path to the trained SEI model (.pth file)
    sequences : torch.Tensor
        Input sequences with shape (batch_size, 4, sequence_length)

    Returns
    -------
    torch.Tensor
        Last embedding layer features with shape (batch_size, 960, 16)
    """
    # Initialize model and load state dict
    model = Sei()
    state_dict = torch.load(model_path, map_location="cpu")

    # Remove 'module.model.' prefix from keys if present
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module.model."):
            new_key = key[len("module.model.") :]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    model.eval()

    # Extract embeddings
    with torch.no_grad():
        # Check if model supports return_embeddings parameter
        import inspect

        sig = inspect.signature(model.forward)
        if "return_embeddings" in sig.parameters:
            embeddings = model(sequences, return_embeddings=True)
            # Reshape to keep 3D structure (batch_size, 960, 16)
            embeddings = embeddings.view(embeddings.size(0), 960, -1)
        else:
            # Alternative: manually extract embeddings by modifying forward pass
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
            embeddings = spline_out.view(spline_out.size(0), 960, model._spline_df)

    return embeddings


def process_csv_to_safetensors(csv_path, model_path, output_path, batch_size=32):
    """
    Process CSV file and extract embeddings, save as safetensors.

    Parameters
    ----------
    csv_path : str
        Path to CSV file with sequence data
    model_path : str
        Path to SEI model
    output_path : str
        Path to save safetensors file
    batch_size : int
        Batch size for processing
    """
    print(f"Loading CSV data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Processing {len(df)} sequences...")

    # Initialise storage for all data
    all_embeddings = []
    all_original_combos = []
    all_fluorescence_values = []

    # Process in batches
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_df = df.iloc[i : i + batch_size]

        # Encode sequences
        batch_sequences = []
        for sequence in batch_df["final_sequence"]:
            encoded_seq = encode_sequence(sequence)
            batch_sequences.append(encoded_seq)

        # Stack into batch tensor
        batch_tensor = torch.stack(batch_sequences)

        # Extract embeddings
        embeddings = extract_last_embedding(model_path, batch_tensor)

        # Store results
        all_embeddings.append(embeddings)
        all_original_combos.extend(batch_df["original_combo"].tolist())
        all_fluorescence_values.extend(batch_df["measured_fluorescence"].tolist())

    # Concatenate all embeddings
    final_embeddings = torch.cat(all_embeddings, dim=0)

    print(f"Final embeddings shape: {final_embeddings.shape}")

    # Prepare data dictionary for safetensors
    save_dict = {
        "embeddings": final_embeddings,
        # 'original_combos': [combo.encode('utf-8') for combo in all_original_combos],  # Store as bytes for safetensors
        "measured_fluorescence": torch.tensor(all_fluorescence_values, dtype=torch.float32),
    }

    # Save as safetensors
    print(f"Saving results to {output_path}...")
    save_file(save_dict, output_path)

    print("Processing complete!")
    print(
        f"Saved {len(all_original_combos)} samples "
        f"with embeddings shape {final_embeddings.shape}"
    )


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from CSV sequences")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/lambda/nfs/evo1zelun/sei-framework/embeddings/input_data/Alcantar_2025/processed_chromatin_sequences.csv",
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
        default="/lambda/nfs/evo1zelun/sei-framework/embeddings/output/Alcantar_2025/processed_chromatin_sequences.safetensors",
        help="Output path for safetensors file",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Process the data
    process_csv_to_safetensors(args.csv_path, args.model_path, args.output_path, args.batch_size)


if __name__ == "__main__":
    main()
