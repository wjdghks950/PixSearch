#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
exec torchrun --standalone --nproc_per_node="${NPROC:-1}" "$ROOT/src/build_multimodal_index.py" "$@"
