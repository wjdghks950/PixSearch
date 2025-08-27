#!/bin/bash

# find a random free port
FREE_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

INDEX_DIR=wiki10k_index  # 'wiki6m_index' is the original name

torchrun --nproc_per_node=8 --master_port=${FREE_PORT} ../build_multimodal_index.py \
         --model_str facebook/dinov3-vitb16-pretrain-lvd1689m \
         --input_jsonl /home/jeonghkim/projects/region-mm-retrieve/data/infoseek/Wiki6M_ver_1_0.jsonl \
         --output_dir /home/jeonghkim/projects/region-mm-retrieve/data/infoseek/${INDEX_DIR} \
         --batch_size 8192 \
         --checkpoint_step 50000 \
         --cache_wiki_images \
         --resume
# --batch_size 8192 \
# --batch_size 2048 \
# --print_length_stats \
# --model_str "sentence-transformers/all-mpnet-base-v2" \
# --model_str "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct" \
# --model_str facebook/dinov3-vitb16-pretrain-lvd1689m \
