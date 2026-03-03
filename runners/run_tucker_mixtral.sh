#!/bin/bash
set -euo pipefail

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false


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
#   bash runners/run_tucker_mixtral.sh <ratio> <whiten_type>
# Example:
#   bash runners/run_tucker_mixtral.sh 0.2 input
if [[ $# -lt 2 ]]; then
  echo "Usage: bash runners/run_tucker_mixtral.sh <ratio: 0.2|0.4|0.6> <whiten_type: input|output|both|none>"
  exit 1
fi

RATIO="$1"
WHITEN_TYPE="$2"

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

python src/run_tucker.py \
    --model_path $MODEL_PATH \
    --save_path $SAVE_PATH \
    --whitening_nsamples 256 \
    --cluster_type "$CLUSTER_TYPE" \
    --model_seq_len 2048 \
    --whiten_type "$WHITEN_TYPE" \
    --layers_to_compress ${LAYERS_TO_COMPRESS[@]} \
    --ratio $RATIO \
    --decomposition_method "svd" \
    --run_eval True \
    --ppl_datasets wikitext2 ptb c4 \
    --eval_tasks openbookqa winogrande piqa arc_easy arc_challenge mathqa


