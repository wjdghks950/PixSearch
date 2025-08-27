
#!/bin/bash

# Find a random free port
FREE_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")


# Check if GPU ids are passed as an argument; if not, ask for it.
if [ -z "$1" ]; then
    read -p "Enter GPU IDs (comma-separated for multiple GPUs): " gpu_ids
else
    gpu_ids=$1
fi

# CKPT_TYPE=plum-13b_kld_0_dice_6_v1_partonomy
# CKPT_TYPE=plum-13b_kld_0.1_focal_tversky_8_v1_0shot - Acc. = ?
CKPT_TYPE="plum-13b_kld_0_dice_6_v1_partonomy"  # TODO: This is the checkpoint for zero-shot PLUM!
DATASET_DIR="/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset"
MODEL_CKPT_PATH="/shared/nas2/jk100/partonomy_private/src/models/PLUM/runs/${CKPT_TYPE}/merged_model"
VAL_DATASET="cragmm"  # pope, gqa, textvqa

export OPENAI_API_KEY="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"

deepspeed --include localhost:${gpu_ids} --master_port=${FREE_PORT} validate_vqa.py \
	  --version=${MODEL_CKPT_PATH} \
	  --dataset_dir=${DATASET_DIR} \
	  --val_dataset="${VAL_DATASET}|val" \
	  --predictions_path=/shared/nas2/jk100/partonomy_private/results/${VAL_DATASET}/${CKPT_TYPE} \
	  --llm_model="gpt-4o" \
	  --use_llm_as_judge \
	  --exp_name=${CKPT_TYPE}

# --version="xinlai/LISA-13B-llama2-v1" \