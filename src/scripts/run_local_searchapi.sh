python ../local_search_api.py \
  --base-dir /home/jeonghkim/projects/pixar/data/infoseek \
  --collection wiki10k_index \
  --image_encoder facebook/dinov3-vitb16-pretrain-lvd1689m \
  --text_encoder sentence-transformers/all-mpnet-base-v2 \
  --host 0.0.0.0 --port 7000 \
  --gpu-index shard \
  --device cuda
