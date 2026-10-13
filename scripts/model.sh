#!/usr/bin/env bash
# Stable entry point: bash scripts/model.sh train|eval|merge|chat|seg [arguments]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
action=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
# Legacy model modules import src.data from pixar, while retrieval imports repo src.
export PYTHONPATH="$ROOT/pixar:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT/pixar/src/model"
case "$action" in
  train) exec deepspeed --include "localhost:${GPUS:-0}" --master_port "${MASTER_PORT:-29500}" pixar_train_ds.py "$@" ;;
  eval) exec python validate_vqa.py "$@" ;;
  merge) exec python merge_lora_weights_and_save_hf_model.py "$@" ;;
  chat) exec python chat_pixar_search.py "$@" ;;
  seg) exec python validate_seg.py "$@" ;;
  *) echo 'Usage: bash scripts/model.sh {train|eval|merge|chat|seg} [Python CLI arguments]'; [[ "$action" == help ]] ;;
esac
