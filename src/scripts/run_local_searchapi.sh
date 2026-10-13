#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
exec python "$ROOT/src/local_search_api.py" \
  --image_encoder facebook/dinov3-vitb16-pretrain-lvd1689m \
  --text_encoder sentence-transformers/all-mpnet-base-v2 \
  --gpu-index none "$@"
