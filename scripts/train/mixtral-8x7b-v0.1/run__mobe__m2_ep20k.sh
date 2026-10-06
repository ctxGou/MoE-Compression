#!/usr/bin/env bash
# MoBE baseline on Mixtral-8x7B-v0.1, m=2 basis matrices.
# Compresses gate_proj and up_proj only (matching MoBE paper original convention).
# Transposed view: rows=4096 (hidden), cols=14336 (ffn), trunc=4096.
#   - gate/up are transposed at load time inside train.py model_variant=mixtral
# One wandb run per (layer, projection); --fast (bf16 + torch.compile) reused across layers
# in the same bash invocation since the inductor cache persists.
# Usage: bash <script> <start_layer> <end_layer_inclusive> <gpu> <gate|up|all>
set -e
START_LAYER=${1:?'Usage: $0 <start_layer> <end_layer_inclusive> <gpu> <gate|up|all>'}
END_LAYER=${2:?}
GPU=${3:?}
PROJ=${4:?}

cd "$(dirname "$0")/../../.."
PYTHON="${PYTHON:-python}"

COMMON="--index_path local_models/Mixtral-8x7B-v0.1/model.safetensors.index.json
  --base_dir local_models/Mixtral-8x7B-v0.1
  --save_path results/wab/mixtral/mobe_m2_nb2_ep20k_lr03
  --num_hidden_layers 32
  --num_matrices 8 --rows_per_matrix 4096 --cols 14336 --truncation 4096
  --num_B 2 --batch_size 8 --num_batches 1
  --learning_rate 0.03 --num_epochs 20000
  --activation silu --model_variant mixtral
  --fast --wandb
  --wandb_project mobe-runs
  --wandb_group mixtral_mobe_m2_ep20k"

# Logging is opt-in; --wandb_mode disabled does not require the wandb package.
COMMON="$COMMON --wandb_mode ${WANDB_MODE:-disabled}"

case "$PROJ" in
  gate) MTYPES="gate_proj" ;;
  up)   MTYPES="up_proj" ;;
  all)  MTYPES="gate_proj up_proj" ;;
  *)    MTYPES="$PROJ" ;;
esac

for LAYER in $(seq ${START_LAYER} ${END_LAYER}); do
  for MTYPE in ${MTYPES}; do
    CUDA_VISIBLE_DEVICES=${GPU} "$PYTHON" train.py $COMMON \
      --start_layer ${LAYER} --end_layer $((LAYER + 1)) \
      --matrix_type ${MTYPE} \
      --wandb_run_name "mixtral_mobe_m2_20k_layer${LAYER}_${MTYPE}"
  done
done
