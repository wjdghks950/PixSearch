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
DATASET_DIR=/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset
MODEL_CKPT_PATH=./runs/${CKPT_TYPE}/merged_model
RESULTS_PATH=../../results
VAL_DATASET=musique # nq, musique, triviaqa, hotpotqa, popqa

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in your environment}"

deepspeed --include localhost:${gpu_ids} --master_port=${FREE_PORT} validate_qa.py \
	  --version=${MODEL_CKPT_PATH} \
	  --model_max_length 512 \
	  --dataset_dir=${DATASET_DIR} \
	  --val_dataset="${VAL_DATASET}|validation" \
	  --predictions_path=${RESULTS_PATH}/${VAL_DATASET}/${CKPT_TYPE} \
	  --retrieve_type title \
	  --limit_batches 1000 \
	  --exp_name=${CKPT_TYPE} \
	  --no_search
	#   --conv_type pixar_0shot_v1

# for --val_dataset="cragmm" do --llm_model="gpt-4.1-mini" and --use_llm_as_judge \
