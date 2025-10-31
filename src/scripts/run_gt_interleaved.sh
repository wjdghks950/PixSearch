python ../construct_gt_interleaved.py \
  --in /home/jeonghkim/projects/pixar/data/region_rag_vqa/webqa/webqa_retrieve_k_5_interleaved_v2.jsonl \
  --out /home/jeonghkim/projects/pixar/data/region_rag_vqa/webqa/webqa_retrieve_k_5_gt_interleaved_v2.jsonl \
  --k 3 \
  --seed 42 \
  --force_style mixed \
  --model gpt-4.1-mini  # override if needed

 # --force_style - mixed | titles | title_content
