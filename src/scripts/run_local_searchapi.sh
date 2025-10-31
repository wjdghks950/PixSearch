#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_local_searchapi.sh --gpus 5,6 --primary 0

# Defaults
GPUS=""                 # e.g., "5,6"
PRIMARY=0               # index within the masked list (0 -> first visible GPU)
GPU_INDEX="shard"       # faiss mode: none|replicate|shard
HOST="0.0.0.0"
PORT=7000

# Parse args
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)        GPUS="$2"; shift 2;;
    --primary)     PRIMARY="$2"; shift 2;;         # 0 or 1 (index in masked list)
    --gpu-index)   GPU_INDEX="$2"; shift 2;;       # none|replicate|shard
    --host)        HOST="$2"; shift 2;;
    --port)        PORT="$2"; shift 2;;
    *)             ARGS+=("$1"); shift;;
  esac
done

# Validate & export mask
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ -n "${GPUS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPUS}"
  IFS=',' read -r -a ARR <<< "${GPUS}"
  if (( PRIMARY < 0 || PRIMARY >= ${#ARR[@]} )); then
    echo "Error: --primary=${PRIMARY} is out of range for --gpus=${GPUS}" >&2
    exit 1
  fi
  DEVICE_ARG="cuda:${PRIMARY}"   # e.g., cuda:0 (maps to physical GPU ${ARR[PRIMARY]})
else
  # No mask given -> use default cuda
  DEVICE_ARG="cuda"
fi

# Launch
python ../local_search_api.py \
  --base-dir /home/jeonghkim/projects/pixar/data/infoseek \
  --collection wiki10k_index \
  --image_encoder facebook/dinov3-vitb16-pretrain-lvd1689m \
  --text_encoder sentence-transformers/all-mpnet-base-v2 \
  --host "${HOST}" --port "${PORT}" \
  --gpu-index "${GPU_INDEX}" \
  --device "${DEVICE_ARG}" \
  "${ARGS[@]}"
