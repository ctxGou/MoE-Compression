#!/usr/bin/env bash
# Evaluate a Mixtral-8x7B-v0.1 (base) checkpoint on the §5.2 prior-comparison suite.
#
# Tasks (all loglikelihood / loglikelihood_rolling — no chat template, no generation):
#     arc_challenge, arc_easy, hellaswag, openbookqa, rte, winogrande, piqa, wikitext
#
# Local task YAMLs preserve the recorded evaluation templates and dataset IDs.
#
# Usage:
#     bash run_eval.sh <model_path> [name] [gpus]
#     model_path : local dir or HF id
#     name       : run name for output dir (default: basename of model_path)
#     gpus       : CUDA_VISIBLE_DEVICES (default: 0,1,2,3); TP = num GPUs
set -euo pipefail
cd "$(dirname "$0")/../../.."
# Assumes `lm_eval` (lm-evaluation-harness v0.4.11) is on PATH.

MODEL=${1:?'Usage: bash run_eval.sh <model_path> [name] [gpus]'}
NAME=${2:-$(basename "$MODEL")}
GPUS=${3:-0,1,2,3}

export CUDA_VISIBLE_DEVICES="$GPUS"
TP=$(awk -F, '{print NF}' <<< "$GPUS")

TS=$(date +%Y%m%d_%H%M%S)
OUT=results/eval/mixtral_${NAME}_${TS}
LOG=logs/eval_mixtral_${NAME}_${TS}.log
mkdir -p "$OUT" logs

echo "Model       : $MODEL"
echo "Name        : $NAME"
echo "GPUs        : $GPUS  (TP=$TP)"
echo "Output      : $OUT"

TASKS="arc_challenge,arc_easy,hellaswag,openbookqa,rte,winogrande,piqa,wikitext"

lm_eval \
    --include_path scripts/eval/mixtral/tasks \
    --model vllm \
    --model_args "pretrained=$MODEL,tensor_parallel_size=$TP,dtype=bfloat16,gpu_memory_utilization=0.80,max_model_len=2048,trust_remote_code=True" \
    --tasks "$TASKS" \
    --num_fewshot 0 \
    --seed 0,1234,1234,1234 \
    --batch_size auto \
    --output_path "$OUT" \
    2>&1 | tee "$LOG"

echo "=== ALL DONE: $NAME ==="
echo "Results : $OUT"
