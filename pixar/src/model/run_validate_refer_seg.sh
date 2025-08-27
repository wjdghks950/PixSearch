#!/bin/bash

# Script for validating PLUM on the Referring Segmentation validation datasets

if [ -z "$1" ]; then
    read -p "Enter GPU IDs (comma-separated for multiple GPUs): " gpu_ids
else
    gpu_ids=$1
fi

# MODEL_TYPE=plum-13b_kld_0.1_focal_tversky_8_v1_partonomy_ft
# MODEL_TYPE=plum-13b_kld_0_dice_6_v1_partonomy
# MODEL_TYPE=plum-13b_kld_0.1_focal_tversky_8_v1_0shot_w_reasonseg
MODEL_TYPE=plum-13b_kld_0.1_focal_tversky_8_v1_0shot_w_reasonseg
MERGED_WEIGHT_DIR=merged_model_9551
MODEL_CKPT_PATH=/shared/nas2/jk100/partonomy_private/src/models/PLUM/runs/${MODEL_TYPE}/${MERGED_WEIGHT_DIR}
VISION_MODEL_PATH=/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/weights/sam_vit_h_4b8939.pth

# referring expression datasets are loaded as: refer_api = REFER(self.base_image_dir, ds, splitBy)

CUDA_VISIBLE_DEVICES=$gpu_ids python validate_seg.py \
	  --version=$MODEL_CKPT_PATH \
	  --val_dataset="refcoco|unc|val" \
	  --dataset_path=/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset/refer_seg \
	  --output_predictions \
	  --predictions_path=/shared/nas2/jk100/partonomy_private/results/${MODEL_TYPE}/refer_seg/predictions.json \
	  --vision_pretrained=$VISION_MODEL_PATH \
	  --exp_name="${MODEL_TYPE}" \
	  --precision='bf16' \
	  --use_bidir_bio \
	  --use_feedback_loop \
	  --eval_only

	# --val_dataset="refcoco|unc|testB" \
	# --val_dataset="refcoco|unc|val" \
	# --val_dataset="refcocog|google|test" \
	# --val_dataset="refcocog|google|val" \
	# --val_dataset="refcocog|umd|test" \
	# --val_dataset="refcocog|umd|val" \
	# --limit_batches=10 \