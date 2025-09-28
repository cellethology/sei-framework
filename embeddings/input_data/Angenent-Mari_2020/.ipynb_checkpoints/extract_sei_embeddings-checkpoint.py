import argparse
import gc
import math
import os
import signal
import sys
import traceback
from datetime import timedelta
from typing import Optional

import pandas as pd
import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.sei import Sei

#NOTE: Need to give in a different sequence length

def get_memory_info():
    """Get current memory usage information"""
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_cached = torch.cuda.memory_reserved() / 1024**3
    else:
        gpu_memory = gpu_allocated = gpu_cached = 0

    ram_info = psutil.virtual_memory()
    ram_total = ram_info.total / 1024**3
    ram_used = ram_info.used / 1024**3

    return {
        'gpu_total': gpu_memory,
        'gpu_allocated': gpu_allocated,
        'gpu_cached': gpu_cached,
        'ram_total': ram_total,
        'ram_used': ram_used
    }

def signal_handler(signum, frame):
    """Handle signals gracefully"""
    print(f"Received signal {signum}, cleaning up...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except:
            pass
    sys.exit(0)


def setup_single_gpu(rank: int, world_size: int, master_port: str = "12355") -> None:
    """Initialize distributed processing for single GPU per process."""
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = master_port
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)

        # Set CUDA device
        torch.cuda.set_device(rank)

        # Clear any existing CUDA cache
        torch.cuda.empty_cache()

        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=30)
        )

        print(f"Rank {rank}: Process group initialized successfully")

    except Exception as e:
        print(f"Rank {rank}: Failed to setup distributed processing: {e}")
        raise

def cleanup() -> None:
    """Clean up distributed processing."""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error during cleanup: {e}")

def worker_main(rank: int, world_size: int, csv_path: str, output_dir: str, batch_size: int, model_path: str) -> None:
    """Global worker function that can be pickled."""

    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    # Load and preprocess data
    print("Setting global max length")
    df = pd.read_csv(csv_path)
    max_sequence_length = df['sequence'].str.len().max()
    print(f"Setting global max length value {max_sequence_length}")

    try:
        # Set up distributed processing
        setup_single_gpu(rank, world_size)

        # Log Initial memory state
        mem_info = get_memory_info()
        print(f"Rank {rank}: Initial memory - GPU: {mem_info['gpu_allocated']:.2f}GB/{mem_info['gpu_total']:.2f}GB, RAM: {mem_info['ram_used']:.2f}GB/{mem_info['ram_total']:.2f}GB")

        # Create output directory
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)

        # Synchronize after directory creation
        # the world_size parameter is the total number of processes participating in the distributed job.
        if world_size > 1:
            dist.barrier()

        # Load SEI
        print(f"Rank {rank}: Loading SEI model")
            # Initialize model and load state dict
        try:
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
        except Exception as e:
            print(f"Rank {rank}: Error loading SEI model: {e}")
            cleanup()
            raise

        # Read CSV data - each rank processes different chunks
        print(f"Rank {rank}: Reading CSV data...")
        try:
            df = pd.read_csv(csv_path)
            total_rows = len(df)

            # Calculate chunk for this rank
            chunk_size = math.ceil(total_rows / world_size)
            start_idx = rank * chunk_size
            end_idx = min(start_idx + chunk_size, total_rows)

            rank_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
            print(f"Rank {rank}: Processing {len(rank_df)} rows (indices {start_idx}:{end_idx})")

            if len(rank_df) == 0:
                print(f"Rank {rank}: No data to process, exiting...")
                cleanup()
                return
        except Exception as e:
            print(f"Rank {rank}: Error reading CSV data: {e}")
            cleanup()
            raise
    except Exception as e:
        print(f"Rank {rank}: Error in worker_main: {e}")
        raise

    # Initialise storage for all data
    all_embeddings = []
    all_expressions = []
    print(f"This is the max_sequence_length {max_sequence_length}")


    effective_batch_size = min(batch_size, 32)
    num_batches = math.ceil(len(rank_df) / effective_batch_size)

    print(f"Rank {rank}: Processing {num_batches} batches with batch size {effective_batch_size}")

    for batch_idx in tqdm(range(num_batches), desc=f"Rank {rank}"):
        try:
            batch_start = batch_idx * effective_batch_size
            batch_end = min(batch_start + effective_batch_size, len(rank_df))
            batch_df = rank_df.iloc[batch_start:batch_end]

            # Encode sequences in batch
            batch_sequences = batch_df['sequence'].tolist()
            batch_encoded_sequences = torch.stack([
                encode_sequence(seq, max_sequence_length) for seq in batch_sequences
            ])

            try:
                # Get embeddings
                with torch.no_grad():
                    batch_embeddings = extract_last_embedding(model_path, batch_encoded_sequences)

                # Store results
                all_embeddings.extend(batch_embeddings.cpu())
                all_expressions.extend(batch_df['expression'].values)

                # Skip empty batches
                if not batch_sequences:
                    continue

                # Memory check before processing
                if batch_idx % 10 == 0:  # Check every 10 batches
                    mem_info = get_memory_info()
                    if mem_info['gpu_allocated'] > mem_info['gpu_total'] * 0.85:  # 85% threshold
                        print(f"Rank {rank}: High GPU memory usage detected, forcing cleanup")
                        torch.cuda.empty_cache()
                        gc.collect()
            except torch.cuda.OutOfMemoryError as e:
                print(f"Rank {rank}: CUDA OOM in batch {batch_idx}, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            except Exception as e:
                print(f"Rank {rank}: Error processing batch {batch_idx}: {e}")
                continue
            finally:
                # Always clear GPU memory after each batch
                if 'batch_embeddings' in locals():
                    del batch_embeddings
                if 'batch_encoded_sequences' in locals():
                    del batch_encoded_sequences
                if 'batch_df' in locals():
                    del batch_df
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Rank {rank}: Fatal error in batch {batch_idx}: {e}")
            traceback.print_exc()
            continue

    if all_embeddings:
        print(f"Rank {rank}: Saving {len(all_embeddings)} sequences...")

        try:
            # Stack embeddings
            final_embeddings = torch.stack(all_embeddings)
            final_expressions = torch.tensor(all_expressions, dtype=torch.float32)

            # Save complete data for this rank
            csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
            rank_output_path = os.path.join(output_dir, f'{csv_basename}_rank_{rank}.safetensors')

            save_data = {
                'embeddings': final_embeddings,
                'expressions': final_expressions
            }

            save_file(save_data, rank_output_path)

            print(f"Rank {rank}: Successfully saved {len(all_expressions)} sequences")

        except Exception as e:
            print(f"Rank {rank}: Error saving results: {e}")
            traceback.print_exc()
    else:
        print(f"Rank {rank}: No embeddings to save")


     # Clean up
    try:
        del model
        torch.cuda.empty_cache()
        gc.collect()
    except:
        pass

    # Synchronize before cleanup
    if world_size > 1:
        print(f"Rank {rank}: Processing complete, waiting for other ranks...")
        dist.barrier()
    print(f"Rank {rank}: All ranks synchronized, exiting...")


def encode_sequence(
    sequence: str,
    sequence_length: int = 4096,
    pad_side: str = "right",   # "right" | "left" | "center"
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    One-hot encode a DNA sequence to a fixed length with explicit padding.
    Unknown bases (e.g., N) are treated as zeros.

    Returns:
        torch.Tensor: shape (4, sequence_length)
    """
    base_to_index = {"A": 0, "T": 1, "G": 2, "C": 3}

    s = (sequence or "").upper()
    L = min(len(s), sequence_length)  # truncate if longer

    # Pre-allocate full-length zero tensor = padding
    encoded = torch.zeros(4, sequence_length, dtype=dtype, device=device)

    # Decide where to place the sequence (pad the rest with zeros)
    if pad_side == "right":
        offset = 0
    elif pad_side == "left":
        offset = sequence_length - L
    elif pad_side == "center":
        offset = (sequence_length - L) // 2
    else:
        raise ValueError("pad_side must be 'right', 'left', or 'center'")

    # Fill one-hot for the (possibly truncated) portion
    for i in range(L):
        idx = base_to_index.get(s[i])
        if idx is not None:
            encoded[idx, i + offset] = 1.0

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
            print("contains return embeddings")
        else:
            print(f"Manual Extraction shape {sequences.shape}")
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
            print("extract last layer embeddings")
    del model
    torch.cuda.empty_cache()
    gc.collect()
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
    all_expressions = []
    max_sequence_length = df['sequence'].str.len().max()
    print(f"This is the max_sequence_length {max_sequence_length}")

    # Process in batches
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_df = df.iloc[i:i+batch_size]

        # Encode sequences
        batch_sequences = []
        for sequence in batch_df['sequence']:
            encoded_seq = encode_sequence(sequence, sequence_length=max_sequence_length)
            batch_sequences.append(encoded_seq)

        # Stack into batch tensor
        batch_tensor = torch.stack(batch_sequences)

        # Extract embeddings
        embeddings = extract_last_embedding(model_path, batch_tensor)

        # Store results
        all_embeddings.append(embeddings)
        all_expressions.extend(batch_df['expression'].tolist())

    # Concatenate all embeddings
    final_embeddings = torch.cat(all_embeddings, dim=0)

    print(f"Final embeddings shape: {final_embeddings.shape}")

    # Prepare data dictionary for safetensors
    save_dict = {
        'embeddings': final_embeddings,
        # 'original_combos': [combo.encode('utf-8') for combo in all_original_combos],  # Store as bytes for safetensors
        'expressions': torch.tensor(all_expressions, dtype=torch.float32),
    }

    # Save as safetensors
    print(f"Saving results to {output_path}...")
    save_file(save_dict, output_path)

    print("Processing complete!")
    print(f"Saved {len(all_expressions)} samples with embeddings shape {final_embeddings.shape}")

def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from CSV sequences')
    parser.add_argument('--csv_path', type=str,
                       default='/lambda/nfs/evo1zelun/sei-framework/embeddings/input_data/Angenent-Mari_2020/expression_clean_off_only.csv',
                       help='Path to CSV file')
    parser.add_argument('--model_path', type=str,
                       default='/lambda/nfs/evo1zelun/sei-framework/model/sei.pth',
                       help='Path to SEI model')
    parser.add_argument('--output_path', type=str,
                       default='/lambda/nfs/evo1zelun/sei-framework/embeddings/output/Angenent-Mari_2020/Angenent-Mari_2020_OFF_ONLY.safetensors',
                       help='Output path for safetensors file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Validate paths
    if not os.path.exists(args.csv_path):
        parser.error(f"Input CSV file not found: {args.csv_path}")

    # Check CUDA availability
    if not torch.cuda.is_available():
        parser.error("CUDA is not available. This script requires GPU support.")

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        parser.error("No GPUs found.")

    print(f"Launching with {n_gpus} GPUs")
    print(f"Processing file: {args.csv_path}")
    print(f"Output directory: {args.output_path}")
    print(f"Batch size: {args.batch_size}")

    # Launch distributed processes with better error handling
    try:
        mp.spawn(
            worker_main,
            args=(n_gpus, args.csv_path, args.output_path, args.batch_size, args.model_path),
            nprocs=n_gpus,
            join=True
        )
        print("All processes completed successfully!")

    except Exception as e:
        print(f"Error in multiprocessing: {e}")
        traceback.print_exc()


    # Process the data
    # process_csv_to_safetensors(args.csv_path, args.model_path, args.output_path, args.batch_size)

if __name__ == "__main__":
    main()
