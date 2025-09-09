#!/bin/bash

# Script for validating PLUM on the Semantic Segmentation validation datasets

if [ -z "$1" ]; then
    read -p "Enter GPU IDs (comma-separated for multiple GPUs): " gpu_ids
else
    gpu_ids=$1
fi

MODEL_TYPE=plum-13b_kld_0.1_focal_tversky_8_v1_0shot_w_reasonseg
# MODEL_TYPE=pixar-13b_kld_0.1_focal_tversky_8_pixar_ft
MERGED_WEIGHT_DIR=merged_model
MODEL_CKPT_PATH=./runs/${MODEL_TYPE}/${MERGED_WEIGHT_DIR}

VISION_TOWER=/home/jeonghkim/projects/pixar/hf_cache/clip-vit-large-patch14
VISION_MODEL_PATH=/home/jeonghkim/projects/pixar/weights/sam_vit_h_4b8939.pth
LLAVA_MODEL_PATH=/home/jeonghkim/projects/pixar/hf_cache/llava-llama-2-13b-chat-lightning-preview

DATASET_DIR=/home/jeonghkim/projects/pixar/data/region_rag_vqa
DATASET_NAME=ade20k  # validation
# DATASET_NAME=cocostuff  # validation
# DATASET_NAME=mapillary  # validation
# DATASET_NAME=paco_lvis  # validation
# DATASET_NAME=pascal_part  # validation

CUDA_VISIBLE_DEVICES=$gpu_ids python validate_seg.py \
	  --version=$MODEL_CKPT_PATH \
	  --backbone=$LLAVA_MODEL_PATH \
	  --vision-tower=$VISION_TOWER \
	  --val_dataset="${DATASET_NAME}|validation" \
	  --dataset_path=$DATASET_DIR \
	  --output_predictions \
	  --predictions_path=/home/jeonghkim/projects/pixar/pixar/src/results/${MODEL_TYPE}/${DATASET_NAME}/predictions.json \
	  --vision_pretrained=$VISION_MODEL_PATH \
	  --exp_name="${MODEL_TYPE}" \
	  --precision='bf16' \
	  --use_bidir_bio \
	  --use_feedback_loop \
	  --eval_only

	# --val_dataset="cocostuff|unc|testB" \
	# --val_dataset="refcoco|unc|val" \
	# --val_dataset="refcocog|google|test" \
	# --val_dataset="refcocog|google|val" \
	# --val_dataset="refcocog|umd|test" \
	# --val_dataset="refcocog|umd|val" \
	# --limit_batches=10 \
