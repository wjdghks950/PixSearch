#!/bin/bash

# LLAVA_MODEL_PATH=liuhaotian/llava-llama-2-13b-chat-lightning-preview
LLAVA_MODEL_PATH=/home/jeonghkim/projects/pixar/hf_cache/llava-llama-2-13b-chat-lightning-preview
VISION_MODEL_PATH=/home/jeonghkim/projects/pixar/weights/sam_vit_h_4b8939.pth
DATASET_DIR=/home/jeonghkim/projects/pixar/data/region_rag_vqa
VISION_TOWER=/home/jeonghkim/projects/pixar/hf_cache/clip-vit-large-patch14

# CKPT_TYPE=plum-13b_kld_0.1_focal_tversky_8_v1_0shot_w_reasonseg
# CKPT_PATH=plum-13b_kld_0.1_focal_tversky_8_v1_0shot_w_reasonseg_bidirbio_2048_maxlen512_epochs25_bidir_bio_feedback_loop_train_prompt_enc_srates_9_5_5_1_ckpt_model

CKPT_TYPE=pixar-13b_kld_0.1_focal_tversky_8_pixar_ft
CKPT_PATH=pixar-13b_kld_0.1_focal_tversky_8_pixar_ft_bidirbio_2048_maxlen512_epochs3_bidir_bio_feedback_loop_train_prompt_enc_pixar_oven_infoseek_aokvqa_okvqa_textvqa_webqa_srates_2_1_1_2_1_1_ckpt_model

# Convert DeepSpeed checkpoint to PyTorch format
echo "Converting DeepSpeed checkpoint to PyTorch format..."
cd ./runs/${CKPT_TYPE}/${CKPT_PATH} && python zero_to_fp32.py . ../pytorch_model.bin
cd ../../..

echo "Merging LoRA weights..."
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version=$LLAVA_MODEL_PATH \
  --weight="./runs/${CKPT_TYPE}/pytorch_model.bin" \
  --save_path="./runs/${CKPT_TYPE}/merged_model" \
  --vision_pretrained=$VISION_MODEL_PATH \
  --vision-tower=$VISION_TOWER \
  --use_bidir_bio \
  --use_feedback_loop \
  --train_mask_prompt_encoder \
  --train_mask_decoder \
  --bidir_nhead=8 \
  --bidir_dim_feedforward=2048 \
  --ce_loss_weight=1.0 \
  --dice_type="focal_tversky" \
  --dice_loss_weight=8 \
  --dice_scale_factor=1000.0 \
  --bce_loss_weight=2.0 \
  --seg_cls_loss_weight=8 \
  --kld_loss_weight=0.1 \
  --kld_sigma=1.0 \
  --focal_tversky_alpha=0.7 \
  --focal_tversky_beta=0.3 \
  --out_dim=256 \
  --model_max_length=512 \
  --lora_r=8 \
  --lora_alpha=16 \
  --lora_dropout=0.05 \
  --lora_target_modules="q_proj,v_proj" \
  --use_mm_start_end

echo "Merge completed! Now you can compare the fixed merged model."
