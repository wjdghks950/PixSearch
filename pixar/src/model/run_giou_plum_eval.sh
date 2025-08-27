#!/bin/bash

# Find a random free port
FREE_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

# 'run_train_plum_ft.sh' uses Partonomy dataset to finetune the models

# 1) Model size
if [ -z "$1" ]; then
    read -p "Enter PLUM model size (13B or 7B): " model_size
else
    model_size=$1
fi

# 2) kld_loss_weight
if [ -z "$2" ]; then
    read -p "Enter kld_loss_weight: " kld_loss_weight
else
    kld_loss_weight=$2
fi

# 3) dice_loss_weight
if [ -z "$3" ]; then
    read -p "Enter dice_loss_weight: " dice_loss_weight
else
    dice_loss_weight=$3
fi

# 4) seg_cls_loss_weight
if [ -z "$4" ]; then
    read -p "Enter seg_cls_loss_weight: " seg_cls_loss_weight
else
    seg_cls_loss_weight=$4
fi

# 5) dice_type
if [ -z "$5" ]; then
    read -p "Use Focal-Tversky Loss? (true/false): " dice_type
else
    dice_type=$5
fi

# 7) BATCH_SIZE
if [ -z "$6" ]; then
    read -p "Enter per device batch size: " BATCH_SIZE
else
    BATCH_SIZE=$6
fi

# 8) precision
if [ -z "$7" ]; then
    read -p "Enable precision? (fp32/bf16/fp16): " precision
else
    precision=$7
fi

# 9) gpu_ids
if [ -z "$8" ]; then
    read -p "Enter GPU IDs (comma-separated for multiple GPUs): " gpu_ids
else
    gpu_ids=$8
fi

# --- NEW: FOCAL_ALPHA and FOCAL_BETA as command-line or prompted input ---
DICE_TYPE="focal_tversky"
# If you're not using focal tversky, you can set defaults or leave empty
FOCAL_ALPHA=0.7
FOCAL_BETA=0.3


# 10) partonomy_dataset_split_name
if [ -z "${9}" ]; then
    read -p "Enter partonomy_dataset_split_name (partimagenet|paco_lvis|pascal_part|partonomy): " partonomy_dataset_split_name
    PARTONOMY_DATASET_SPLIT_NAME=${partonomy_dataset_split_name:-"partimagenet"}
else
    PARTONOMY_DATASET_SPLIT_NAME=${9}
fi

LLAVA_MODEL_PATH="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
# LLAVA_MODEL_PATH="/shared/nas2/jk100/partonomy_private/weights/liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview"

VISION_MODEL_PATH=/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/weights/sam_vit_h_4b8939.pth
DATASET_DIR=/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset

# TODO: Change ZERO_SHOT_CKPT_PATH to the path of the zero-shot checkpoint after training with `run_train_plum_0shot.sh`
# ZERO_SHOT_CKPT_PATH="runs/plum-13b_kld_0.1_focal_tversky_8_v1_0shot/plum-13b_kld_0.1_focal_tversky_8_v1_0shot_bidirbio_2048_accum_10_maxlen512_epochs25_segloss_2_bce_loss_2_kld_loss_0_dice_loss_8_bidir_bio_train_prompt_enc_ckpt_model"
# EXP_NAME="plum-${model_size}_kld_${kld_loss_weight}_${DICE_TYPE}_${dice_loss_weight}_v1_partonomy_ft_focal_tversky_04_06"
ZERO_SHOT_CKPT_PATH="runs/plum-13b_kld_0.1_focal_tversky_8_v1_partonomy_ft/plum-13b_kld_0.1_focal_tversky_8_v1_partonomy_ft_bidirbio_2048_maxlen512_epochs20_bidir_bio_feedback_loop_exp_seg_train_prompt_enc_exp_seg_val_partonomy_partimagenet_paco_lvis_pascal_part_ckpt_model"
EXP_NAME="plum-${model_size}_kld_${kld_loss_weight}_${DICE_TYPE}_${dice_loss_weight}_v1_partonomy_ft"


echo "Running experiment: $EXP_NAME"
echo " - kld_loss_weight=${kld_loss_weight}"
echo " - dice_loss_weight=${dice_loss_weight}"
echo " - teacher_ref_input=${teacher_ref_input}"
echo " - DICE_TYPE=${DICE_TYPE}"
echo " - FOCAL_ALPHA=${FOCAL_ALPHA}"
echo " - FOCAL_BETA=${FOCAL_BETA}"
echo " - Using GPUs: ${gpu_ids}"
echo " - Using port: ${FREE_PORT}"

deepspeed --include localhost:${gpu_ids} --master_port=${FREE_PORT} plum_train_ds.py \
    --version=$LLAVA_MODEL_PATH \
    --dataset_dir=$DATASET_DIR \
    --vision_pretrained=$VISION_MODEL_PATH \
    --zero_shot_ckpt_path=$ZERO_SHOT_CKPT_PATH \
    --dataset="explanatory_seg" \
    --explanatory_seg_data \
    --partonomy_dataset_split=$PARTONOMY_DATASET_SPLIT_NAME \
    --sample_one_question_per_image \
    --random_seed=42 \
    --sample_rates="1" \
    --batch_size=$BATCH_SIZE \
    --grad_accumulation_steps 10 \
    --ce_loss_weight=1.0 \
    --dice_type=${DICE_TYPE} \
    --dice_loss_weight=${dice_loss_weight} \
    --bce_loss_weight=2.0 \
    --kld_loss_weight=${kld_loss_weight} \
    --seg_cls_loss_weight=${seg_cls_loss_weight} \
    --exp_name=$EXP_NAME \
    --precision=$precision \
    --model_max_length 512 \
    --dice_scale_factor 1000.0 \
    --epochs 5 \
    --focal_tversky_alpha=$FOCAL_ALPHA \
    --focal_tversky_beta=$FOCAL_BETA \
    --bidir_dim_feedforward 2048 \
    --lr 3e-4 \
    --train_mask_prompt_encoder \
    --use_feedback_loop \
    --use_bidir_bio \
    --eval_only \
    --output_predictions \
    --auto_resume \
    --debug