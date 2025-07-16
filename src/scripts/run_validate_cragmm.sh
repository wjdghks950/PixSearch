#!/bin/bash

python ../validate_cragmm.py \
    --version "/home/jeonghkim/weights/meta-llama/Llama-4-Scout-17B-16E-Instruct" \
    --model "Llama-4-Scout-17B-16E-Instruct-FP8" \
    --val_dataset "cragmm" \
    --split "validation" \
    --turn_type "single" \
    --resize_img \
    --use_llm_as_judge \
    --use_crop \
    --use_gold_crop \
    --llm_model gpt-4o-mini \
    --use_retrieved \
    --topk 5 \
    --image_only
    # --text_only \
    # --use_crop
    # --use_llama_api
    # --use_summarized
    # --use_reranked
    # --use_crop \
    # --use_gold_crop \
    # --use_llama_api \
    # --debug
