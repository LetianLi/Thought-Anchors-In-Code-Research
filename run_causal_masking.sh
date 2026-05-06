#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

LOG_DIR="results/causal_masking_logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

run_dataset() {
    local dataset=$1
    local log="$LOG_DIR/${dataset}_${TIMESTAMP}.log"
    echo "=== Starting $dataset at $(date) ===" | tee "$log"
    local start=$SECONDS
    python -m thought_anchors_code.analysis.whitebox_masking.run \
        --dataset "$dataset" \
        --resume \
        2>&1 | tee -a "$log"
    local elapsed=$(( SECONDS - start ))
    echo "=== $dataset done in ${elapsed}s at $(date) ===" | tee -a "$log"
}

run_dataset humaneval
run_dataset mbpp

echo "=== All done at $(date) ==="
