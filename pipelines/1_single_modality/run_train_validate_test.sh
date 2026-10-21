#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
exec "${PYTHON:-python}" -m pipelines.1_single_modality.train_validate_test "$@"
