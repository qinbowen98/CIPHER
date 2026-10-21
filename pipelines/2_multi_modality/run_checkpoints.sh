#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
exec "${PYTHON:-python}" -m pipelines.2_multi_modality.evaluate_checkpoints "$@"
