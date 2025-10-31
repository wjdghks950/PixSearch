#!/bin/bash

# Find a random free port
FREE_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")


# Check if GPU ids are passed as an argument; if not, ask for it.
if [ -z "$1" ]; then
    read -p "Enter GPU IDs (comma-separated for multiple GPUs): " gpu_ids
else
    gpu_ids=$1
fi

# CKPT_TYPE=pixar-13b_kld_0.1_focal_tversky_8_pixar_ft  # <- <information> </information> token is not replaced with <pad>
CKPT_TYPE=pixar-13b_kld_0.1_focal_tversky_8_pixar_ft_info_tok_padmask_lr_3e-4_gt_interleaved  # trained with max_model_len = 512 & <info> tokens replaced with <pad>
# CKPT_TYPE=plum-13b_kld_0.1_focal_tversky_8_v1_0shot_w_reasonseg

# DATASET_DIR="/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset"
DATASET_DIR=/home/jeonghkim/projects/pixar/data/region_rag_vqa
MODEL_CKPT_PATH=./runs/${CKPT_TYPE}/merged_model
RESULTS_PATH=/home/jeonghkim/projects/pixar/pixar/src/results
VAL_DATASET=cragmm  # pope, gqa, textvqa, infoseek, oven, cragmm

export OPENAI_API_KEY="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"

deepspeed --include localhost:${gpu_ids} --master_port=${FREE_PORT} validate_vqa.py \
	  --version=${MODEL_CKPT_PATH} \
	  --model_max_length 512 \
	  --dataset_dir=${DATASET_DIR} \
	  --val_dataset="${VAL_DATASET}|val" \
	  --predictions_path=${RESULTS_PATH}/${VAL_DATASET}/${CKPT_TYPE} \
	  --retrieve_type title \
	  --limit_batches 1000 \
	  --exp_name=${CKPT_TYPE} \
	  --llm_model="gpt-4.1-mini" \
	  --use_llm_as_judge \
	  --search_api cragmm
	#   --conv_type pixar_0shot_v1

# for --val_dataset="cragmm" do --llm_model="gpt-4.1-mini" and --use_llm_as_judge \
