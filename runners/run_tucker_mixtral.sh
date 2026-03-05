#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# 限制 CPU 线程数
export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=18
export NUMEXPR_NUM_THREADS=18
export OPENBLAS_NUM_THREADS=18


SAVE_PATH="./results"
DATASET="wikitext2"
CLUSTER_TYPE="global" # global group mixed 
MODEL_PATH="mistralai/Mixtral-8x7B-v0.1"

# Usage:
#   bash runners/run_tucker_mixtral.sh <ratio> <whiten_type> [ratio_scope]
# Example:
#   bash runners/run_tucker_mixtral.sh 0.2 input global
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash runners/run_tucker_mixtral.sh <ratio: 0.2|0.4|0.6> <whiten_type: input|output|both|none> [ratio_scope: local|global]"
  exit 1
fi

RATIO="$1"
WHITEN_TYPE="$2"
RATIO_SCOPE="${3:-local}"

case "$WHITEN_TYPE" in
  input|output|both|none) ;;
  *)
    echo "Invalid whiten_type: $WHITEN_TYPE"
    echo "Allowed: input | output | both | none"
    exit 1
    ;;
esac

case "$RATIO" in
  0.2)
    LAYERS_TO_COMPRESS=(3 5 6 7 9 12 23 24 25)
    ;;
  0.4)
    LAYERS_TO_COMPRESS=(3 5 6 7 9 10 12 13 21 22 23 24 25 26)
    ;;
  0.6)
    LAYERS_TO_COMPRESS=(2 3 5 6 7 9 10 12 13 14 15 16 17 20 21 22 23 24 25 26 27)
    ;;
  *)
    echo "Invalid ratio: $RATIO"
    echo "Allowed: 0.2 | 0.4 | 0.6"
    exit 1
    ;;
esac

case "$RATIO_SCOPE" in
  local|global) ;;
  *)
    echo "Invalid ratio_scope: $RATIO_SCOPE"
    echo "Allowed: local | global"
    exit 1
    ;;
esac

python src/run_tucker.py \
    --model_path $MODEL_PATH \
    --save_path $SAVE_PATH \
    --whitening_nsamples 256 \
    --cluster_type "$CLUSTER_TYPE" \
    --model_seq_len 2048 \
    --whiten_type "$WHITEN_TYPE" \
    --ratio_scope "$RATIO_SCOPE" \
    --layers_to_compress ${LAYERS_TO_COMPRESS[@]} \
    --ratio $RATIO \
    --decomposition_method "svd" \
    --run_eval True \
    --ppl_datasets wikitext2 ptb c4 \
    --eval_tasks openbookqa winogrande piqa arc_easy arc_challenge mathqa

