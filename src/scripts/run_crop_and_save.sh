#!/bin/bash

python ../crop_cragmm_images.py \
	--output_dir ../../results/cragmm/fixed \
    --sam_checkpoint ~/weights/sam/sam_vit_h_4b8939.pth \
    --sam_model_type vit_h \
    --dino_config ~/weights/groundingdino/GroundingDINO_SwinT_OGC.py \
    --dino_checkpoint ~/weights/groundingdino/groundingdino_swint_ogc.pth \
    --use_llm_as_judge \
    --llm_model gpt-4o
