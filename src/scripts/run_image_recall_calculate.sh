#!/bin/bash

python ../image_recall_calculate.py \
    --use_cached_results \
    --use_split_k \
    --results_output_path "/home/jeonghkim//projects/region-mm-retrieve/results/cragmm/recall_results_bbox_mask.json"
