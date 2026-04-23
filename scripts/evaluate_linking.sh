#!/usr/bin/env bash
# ------------------------------------------------------------------
# KaSLA — evaluate a single schema-linking result file
# ------------------------------------------------------------------
# Usage:
#   bash scripts/evaluate_linking.sh <path/to/linking.json>
# ------------------------------------------------------------------
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <linking_result.json>"
    exit 1
fi

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "${PROJECT_ROOT}"

python -u eval_schema-linking.py --data "$1"
