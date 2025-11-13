import torch
import numpy as np
import pandas as pd
import sys
import os
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
import argparse

# Add parent directory to path for imports
# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
from model.sei import Sei
from embeddings.util import encode_sequence

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
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Remove 'module.model.' prefix from keys if present
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.model.'):
            new_key = key[len('module.model.'):]
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
        if 'return_embeddings' in sig.parameters:
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
    all_variant_ids = []
    all_conc = ['0 mg/L cerulenin', '1 mg/L cerulenin', '2 mg/L cerulenin', '3 mg/L cerulenin', '5 mg/L cerulenin', '8 mg/L cerulenin']
    con_dict = {key: None for key in all_conc}
    
    # Process in batches
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_df = df.iloc[i:i+batch_size]
        
        # Encode sequences
        batch_sequences = []
        for sequence in batch_df['full_sequence']:
            encoded_seq = encode_sequence(sequence)
            batch_sequences.append(encoded_seq)
        
        # Stack into batch tensor
        batch_tensor = torch.stack(batch_sequences)
        
        # Extract embeddings
        embeddings = extract_last_embedding(model_path, batch_tensor)
        
        # Store results
        all_embeddings.append(embeddings)
        all_variant_ids.extend(batch_df['sequence_index'].tolist())
        all_expressions.extend(batch_df['Expression'].tolist())
    
    # Concatenate all embeddings
    final_embeddings = torch.cat(all_embeddings, dim=0)
    
    print(f"Final embeddings shape: {final_embeddings.shape}")
    
    # Prepare data dictionary for safetensors
    save_dict = {
        'embeddings': final_embeddings,
        'variant_ids': torch.tensor(all_variant_ids),
        'expressions': torch.tensor(all_expressions, dtype=torch.float32),
    }
    
    # Save as safetensors
    print(f"Saving results to {output_path}...")
    save_file(save_dict, output_path)
    
    print("Processing complete!")
    print(f"Saved {len(all_variant_ids)} samples with embeddings shape {final_embeddings.shape}")

def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from CSV sequences')
    parser.add_argument('--csv_path', type=str, 
                       default='/lambda/nfs/evo1zelun/sei-framework/embeddings/input_data/all_data_with_sequence.csv',
                       help='Path to CSV file')
    parser.add_argument('--model_path', type=str,
                       default='/lambda/nfs/evo1zelun/sei-framework/model/sei.pth',
                       help='Path to SEI model')
    parser.add_argument('--output_path', type=str,
                       default='/home/ubuntu/evo1zelun/sei-framework/embeddings/output/166k_sei_embeddings.safetensors',
                       help='Output path for safetensors file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Process the data
    process_csv_to_safetensors(args.csv_path, args.model_path, args.output_path, args.batch_size)

if __name__ == "__main__":
    main()