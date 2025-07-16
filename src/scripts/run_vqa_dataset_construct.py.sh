#!/bin/bash

python ../vqa_dataset_construct.py \
    --debug \
    --limit_num 200 \
    --add_incontext_samples \
    --num_incontext_samples 8 \
