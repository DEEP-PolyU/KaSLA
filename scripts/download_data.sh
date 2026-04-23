#!/usr/bin/env bash
# ------------------------------------------------------------------
# KaSLA — download the Google Drive data bundle
# ------------------------------------------------------------------
# Requires: gdown (pip install gdown)
# ------------------------------------------------------------------
set -euo pipefail

FILE_ID="1vv_6ZdOGGsMZ1P0emPIPn4ZE7A8qKNak"
OUT="kasla_data_bundle.zip"

if ! command -v gdown &> /dev/null; then
    echo "[WARN] gdown not found; installing via pip ..."
    pip install --quiet gdown
fi

echo "[KaSLA] Downloading data bundle from Google Drive ..."
gdown --id "${FILE_ID}" -O "${OUT}"

echo "[KaSLA] Extracting into data/ ..."
mkdir -p data
unzip -oq "${OUT}" -d data/
rm -f "${OUT}"

echo "[KaSLA] Done. Contents of data/:"
ls -la data/
