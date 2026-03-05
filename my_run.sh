#!/bin/bash
# set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_DATASETS_TRUST_REMOTE_CODE=1


RUNNER="runners/run_tucker_mixtral.sh"
RATIO_SCOPE="local" # local global

ratios=(0.6)
whiten_types=(input none)

for r in "${ratios[@]}"; do
  for w in "${whiten_types[@]}"; do
    echo ">>> Running ratio=${r}, whiten_type=${w}, ratio_scope=${RATIO_SCOPE}"
    if ! bash "$RUNNER" "$r" "$w" "$RATIO_SCOPE"; then
      echo "FAILED ratio=${r}, whiten_type=${w}, ratio_scope=${RATIO_SCOPE}, continuing..."
    fi
  done
done

echo "All runs finished."
