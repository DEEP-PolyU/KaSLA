#!/usr/bin/env bash
set -e

# Example: build SFT data for BIRD training set
CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
    --sic_path ../sic_ckpts/sic_bird_with_evidence \
    --mode sg-t2sTsl-fullS-d \
    --dataset_path ../data/bird_train_full.json \
    --target_dataset_path <output_path>
