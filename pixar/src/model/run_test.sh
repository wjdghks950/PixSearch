#!/bin/bash

# Check if GPU IDs were provided
if [ -z "$1" ]; then
    echo "Usage: ./occupy_gpu.sh <GPU_IDS> [MEMORY_FRACTION] [BLOCK_SIZE_MB]"
    echo "Example: ./occupy_gpu.sh 0,1,2 0.88 100"
    exit 1
fi

# GPU IDs as a comma-separated string (e.g., "0,1,2")
GPU_IDS=$1
MEMORY_FRACTION=${2:-0.87}  # Default to 88% if not provided
BLOCK_SIZE_MB=${3:-100}     # Default block size is 100MB if not provided

# Run the Python script
python run_test.py --gpu_ids "$GPU_IDS" --fraction "$MEMORY_FRACTION" --block_size "$BLOCK_SIZE_MB"
