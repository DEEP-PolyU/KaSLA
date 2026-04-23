#!/usr/bin/env bash
# ------------------------------------------------------------------
# KaSLA — one-shot quick start on BIRD-dev
# ------------------------------------------------------------------
# This script:
#   1. Runs KaSLA on the default in-context-learning linker output.
#   2. Evaluates all provided baseline + KaSLA linking results.
# ------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "${PROJECT_ROOT}"

echo "[KaSLA] Running knapsack-based schema linking ..."
python -u KaSLA.py

echo "[KaSLA] Evaluating baselines + KaSLA ..."
mkdir -p linking-results
bash eval_schema-linking.sh

echo "[KaSLA] Done."
