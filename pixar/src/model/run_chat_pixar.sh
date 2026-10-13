#!/usr/bin/env bash
# Explicit Python CLI arguments replace the old machine-specific positional recipe.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
exec bash "$ROOT/scripts/model.sh" chat "$@"
