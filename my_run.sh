#!/bin/bash
set -euo pipefail

RUNNER="runners/run_tucker_mixtral.sh"

ratios=(0.4 0.6)
whiten_types=(input none)

for r in "${ratios[@]}"; do
  for w in "${whiten_types[@]}"; do
    echo ">>> Running ratio=${r}, whiten_type=${w}"
    bash "$RUNNER" "$r" "$w"
  done
done

echo "All runs finished."
