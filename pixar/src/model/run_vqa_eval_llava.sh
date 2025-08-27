#!/bin/bash

# Check if GPU ids are passed as an argument; if not, ask for it.
if [ -z "$1" ]; then
    read -p "Enter GPU IDs (comma-separated for multiple GPUs): " gpu_ids
else
    gpu_ids=$1
fi

DATASET_DIR="/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset"
MODEL_VERSION="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
VAL_DATASET="pope"  # pope, gqa, textvqa

deepspeed --include localhost:${gpu_ids} --master_port=24998 validate_vqa_llava.py \
    --version=$MODEL_VERSION \
    --dataset_dir=$DATASET_DIR \
    --val_dataset="${VAL_DATASET}|val" \
    --predictions_path=/shared/nas2/jk100/partonomy_private/results/${VAL_DATASET}/llava_base \
    --exp_name=llava-13b-vqa-eval 