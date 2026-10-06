#!/usr/bin/env bash
# Compatibility wrapper: shell END is inclusive; Python END is exclusive.
# The YAML includes the six final uv_lr_scale=0.05 layer/projection overrides.
set -euo pipefail
START=${1:?'Usage: bash run.sh <start> <end_inclusive> <gpu> <gate|up|gateup|all> [launcher options]'}
END=${2:?}
GPU=${3:?}
PROJ=${4:?}
shift 4
cd "$(dirname "$0")/../../.."
case "$PROJ" in
  gate) PROJECTIONS=(gate_proj) ;;
  up) PROJECTIONS=(up_proj) ;;
  gateup|all) PROJECTIONS=(gate_proj up_proj) ;;
  *) echo "Only gate/up compression is released: $PROJ" >&2; exit 2 ;;
esac
OPTIONS=()
if [[ -n "${BASE_MODEL:-}" ]]; then OPTIONS+=(--base_dir "$BASE_MODEL"); fi
if [[ -n "${WAB_DIR:-}" ]]; then OPTIONS+=(--save_path "$WAB_DIR"); fi
exec "${PYTHON:-python}" scripts/train/from_yaml.py \
  --start_layer "$START" --end_layer "$((END + 1))" --gpu "$GPU" \
  --projections "${PROJECTIONS[@]}" --wandb_mode "${WANDB_MODE:-disabled}" \
  "${OPTIONS[@]}" "$@"
