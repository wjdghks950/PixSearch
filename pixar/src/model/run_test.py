import torch
import time
import argparse
import os

def occupy_gpu_memory(gpu_ids, fraction=0.88, block_size_mb=100):
    """
    Occupy a specified percentage of GPU memory on a given GPU.

    Args:
        gpu_ids (list): List of GPU IDs to allocate memory on.
        fraction (float): The fraction of GPU memory to allocate (default: 0.88).
        block_size_mb (int): The size of each allocation block in MB (default: 100MB).
    """
    for gpu_id in gpu_ids:
        device = torch.device(f"cuda:{gpu_id}")  # Use index 0 within the assigned device
        props = torch.cuda.get_device_properties(device)
        total_mem = props.total_memory  # Total memory in bytes
        target_alloc = int(total_mem * fraction)  # Target allocation in bytes
        block_size_bytes = block_size_mb * 1024 * 1024  # Convert MB to bytes
        bytes_per_float = 4  # float32 size in bytes

        allocated_tensors = []  # Prevents garbage collection

        print(f"GPU {gpu_id}: Total Memory = {total_mem / (1024**2):.2f} MB, Target Allocation = {target_alloc / (1024**2):.2f} MB")
        
        allocated = 0  # Track allocated memory
        while allocated < target_alloc:
            # Determine the next block size without exceeding the target
            alloc_size = block_size_bytes if allocated + block_size_bytes <= target_alloc else target_alloc - allocated
            num_elements = alloc_size // bytes_per_float  # Number of float32 elements

            try:
                tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
                allocated_tensors.append(tensor)
                allocated += alloc_size
                print(f"GPU {gpu_id}: Allocated {allocated / (1024**2):.2f} MB")
            except RuntimeError as e:
                print(f"GPU {gpu_id}: Allocation error: {e}")
                break

    print(f"GPU {gpu_id}: Reached target allocation ({fraction * 100:.0f}% of capacity).\n")

    print("GPU memory is occupied. The script will now continue running. Press Ctrl+C to terminate.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Script terminated by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Occupy GPU memory to prevent other jobs from stealing it.")
    parser.add_argument("--gpu_ids", type=str, required=True, help="GPU ID to occupy memory on.")
    parser.add_argument("--fraction", type=float, default=0.88, help="Fraction of GPU memory to occupy (default: 0.88).")
    parser.add_argument("--block_size", type=int, default=100, help="Block size in MB for allocation (default: 100MB).")
    args = parser.parse_args()
    
    gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(",") if g.strip().isdigit()]

    occupy_gpu_memory(gpu_ids, args.fraction, args.block_size)
