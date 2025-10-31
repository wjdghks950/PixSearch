#!/bin/bash

python ../vqa_dataset_construct.py \
    --version_num 1 \
    --limit_num 7000 \
    --add_incontext_samples \
    --llm_model gpt-4.1-mini \
    --num_incontext_samples 16 \
    --retriever cragmm \
    --use_llm_for_span_extraction \
    --debug
# For non cragmm datasets,set '--retriever local'
