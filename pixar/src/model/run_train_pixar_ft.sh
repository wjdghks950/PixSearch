#!/bin/bash

# ./run_train_pixar_ft.sh 13B 0.1 8 2 false true 8 bf16 5 0.7 0.3

# Find a random free port
FREE_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1

# 'run_train_pixar_ft.sh' uses Pixar dataset to finetune the models

# 1) Model size
if [ -z "$1" ]; then
    read -p "Enter Pixar model size (13B or 7B): " model_size
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

# 5) Teacher reference input
if [ -z "$5" ]; then
    read -p "Enable teacher reference? (true/false): " teacher_ref_input
else
    teacher_ref_input=$5
fi

# 6) dice_type
if [ -z "$6" ]; then
    read -p "Use Focal-Tversky Loss? (true/false): " dice_type
else
    dice_type=$6
fi

# 7) BATCH_SIZE
if [ -z "$7" ]; then
    read -p "Enter per device batch size: " BATCH_SIZE
else
    BATCH_SIZE=$7
fi

# 8) precision
if [ -z "$8" ]; then
    read -p "Enable precision? (fp32/bf16/fp16): " precision
else
    precision=$8
fi

# 9) gpu_ids
if [ -z "$9" ]; then
    read -p "Enter GPU IDs (comma-separated for multiple GPUs): " gpu_ids
else
    gpu_ids=$9
fi

# --- NEW: FOCAL_ALPHA and FOCAL_BETA as command-line or prompted input ---

if [ "$dice_type" == "true" ]; then
    DICE_TYPE="focal_tversky"

    # 10) FOCAL_ALPHA
    if [ -z "${10}" ]; then
        read -p "Enter FOCAL_ALPHA (default 0.7): " input_focal_alpha
        FOCAL_ALPHA=${input_focal_alpha:-0.7}
    else
        FOCAL_ALPHA=${10}
    fi

    # 11) FOCAL_BETA
    if [ -z "${11}" ]; then
        read -p "Enter FOCAL_BETA (default 0.3): " input_focal_beta
        FOCAL_BETA=${input_focal_beta:-0.3}
    else
        FOCAL_BETA=${11}
    fi

else
    DICE_TYPE="dice"
    # If you're not using focal tversky, you can set defaults or leave empty
    FOCAL_ALPHA=0.7
    FOCAL_BETA=0.3
fi

# Teacher reference flag
if [ "$teacher_ref_input" == "true" ]; then
    TEACHER_REF="--use_teacher_ref"
else
    TEACHER_REF=""
fi

# Convert model_size to lower case
model_size=$(echo "$model_size" | tr '[:upper:]' '[:lower:]')

# LLAVA_MODEL_PATH=liuhaotian/llava-llama-2-13b-chat-lightning-preview

if [ "$model_size" = "13b" ]; then
    # LLAVA_MODEL_PATH="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    LLAVA_MODEL_PATH=/home/jeonghkim/projects/pixar/hf_cache/llava-llama-2-13b-chat-lightning-preview
elif [ "$model_size" = "7b" ]; then
    LLAVA_MODEL_PATH=/shared/nas2/jk100/partonomy_private/weights/liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview
else
    echo "Invalid model size. Please use 13B or 7B."
    exit 1
fi

VISION_MODEL_PATH=/home/jeonghkim/projects/pixar/weights/sam_vit_h_4b8939.pth
DATASET_DIR=/home/jeonghkim/projects/pixar/data/region_rag_vqa
VISION_TOWER=/home/jeonghkim/projects/pixar/hf_cache/clip-vit-large-patch14

# TODO: Change ZERO_SHOT_CKPT_PATH to the path of the zero-shot checkpoint after trainin with `run_train_plum_0shot.sh`
# # [Original]
# ZERO_SHOT_CKPT_PATH="runs/plum-13b_kld_0.1_focal_tversky_8_v1_0shot/plum-13b_kld_0.1_focal_tversky_8_v1_0shot_bidirbio_2048_accum_10_maxlen512_epochs25_segloss_2_bce_loss_2_kld_loss_0_dice_loss_8_bidir_bio_train_prompt_enc_ckpt_model"
# # [DICE + Bidirectional Bio]
# ZERO_SHOT_CKPT_PATH="runs/plum-13b_kld_0.1_dice_8_v1_0shot_w_reasonseg/plum-13b_kld_0.1_dice_8_v1_0shot_w_reasonseg_bidirbio_2048_maxlen512_epochs25_bidir_bio_feedback_loop_train_prompt_enc_focal_tversky_04_06_srates_9_5_5_1_ckpt_model"
# # [Focal Tversky + Bidirectional Bio]
ZERO_SHOT_CKPT_PATH="runs/plum-13b_kld_0.1_focal_tversky_8_v1_0shot_w_reasonseg/plum-13b_kld_0.1_focal_tversky_8_v1_0shot_w_reasonseg_bidirbio_2048_maxlen512_epochs25_bidir_bio_feedback_loop_train_prompt_enc_srates_9_5_5_1_ckpt_model"
# # [Focal Tversky - No Bidirectional Bio]
# ZERO_SHOT_CKPT_PATH="runs/plum-13b_kld_0.1_focal_tversky_8_v1_0shot_w_reasonseg/plum-13b_kld_0.1_focal_tversky_8_v1_0shot_w_reasonseg_maxlen512_epochs25_kld_loss_0_dice_loss_8_feedback_loop_train_prompt_enc_srates_9_5_5_1_ckpt_model"
# # [Focal Tversky - No Feedback Loop]
# ZERO_SHOT_CKPT_PATH="runs/plum-13b_kld_0.1_focal_tversky_8_v1_0shot_w_reasonseg/plum-13b_kld_0.1_focal_tversky_8_v1_0shot_w_reasonseg_bidirbio_2048_maxlen512_epochs25_bidir_bio_train_prompt_enc_srates_9_5_5_1_ckpt_model"

EXP_NAME="pixar-${model_size}_kld_${kld_loss_weight}_${DICE_TYPE}_${dice_loss_weight}_pixar_ft_info_tok_padmask_lr_3e-4_gt_interleaved"
# EVAL_CKPT_PATH="runs/plum-13b_kld_0.1_focal_tversky_8_v1_0shot/plum-13b_kld_0.1_focal_tversky_8_v1_0shot_bidirbio_2048_accum_10_maxlen512_epochs25_segloss_2_bce_loss_2_kld_loss_0_dice_loss_8_bidir_bio_train_prompt_enc_ckpt_model"

echo "Running experiment: $EXP_NAME"
echo " - kld_loss_weight=${kld_loss_weight}"
echo " - dice_loss_weight=${dice_loss_weight}"
echo " - teacher_ref_input=${teacher_ref_input}"
echo " - DICE_TYPE=${DICE_TYPE}"
echo " - FOCAL_ALPHA=${FOCAL_ALPHA}"
echo " - FOCAL_BETA=${FOCAL_BETA}"
echo " - Using GPUs: ${gpu_ids}"
echo " - Using port: ${FREE_PORT}"

deepspeed --include localhost:${gpu_ids} --master_port=${FREE_PORT} pixar_train_ds.py \
    --version=$LLAVA_MODEL_PATH \
    --dataset_dir=$DATASET_DIR \
    --vision_pretrained=$VISION_MODEL_PATH \
    --vision-tower=$VISION_TOWER \
    --zero_shot_ckpt_path=$ZERO_SHOT_CKPT_PATH \
    --dataset="sem_seg||refer_seg||vqa||reason_seg||pixar" \
    --pixar_data="oven||infoseek||aokvqa||okvqa||textvqa" \
    --random_seed=42 \
    --sample_rates="3,3,5,1,9" \
    --batch_size=$BATCH_SIZE \
    --grad_accumulation_steps 10 \
    --ce_loss_weight=1.0 \
    --dice_type=${DICE_TYPE} \
    --dice_loss_weight=${dice_loss_weight} \
    --bce_loss_weight=2.0 \
    --kld_loss_weight=${kld_loss_weight} \
    --seg_cls_loss_weight=${seg_cls_loss_weight} \
    --exp_name=$EXP_NAME \
    $TEACHER_REF \
    --precision=$precision \
    --model_max_length 512 \
    --dice_scale_factor 1000.0 \
    --epochs 20 \
    --train_mask_prompt_encoder \
    --focal_tversky_alpha=$FOCAL_ALPHA \
    --focal_tversky_beta=$FOCAL_BETA \
    --use_bidir_bio \
    --bidir_dim_feedforward 2048 \
    --lr 3e-4 \
    --use_feedback_loop \
    --info_mask_type="pad" \
    --auto_resume
    # --conv_type pixar_0shot_v1

# --lr 3e-4  # original lr
# --pixar_data="oven||infoseek||aokvqa||okvqa||textvqa||webqa" \
