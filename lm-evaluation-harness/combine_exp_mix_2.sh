#!/bin/bash
export PYTHONPATH='./lm-evaluation-harness':'/workspace/SVD-MOE-new'
# 全局变量
SVD_MODEL_PATH=""

# 共同参数
# hkust
# MODEL_PATH="./phimoe"

# MODEL_PATH="/workspace/SVD-MOE-new/component/phimoe"
MODEL_PATH="mistralai/Mixtral-8x7B-v0.1"
WHITENING_NSAMPLES=1024
UPDATING_NSAMPLES=16
SEED=42
MODEL_SEQ_LEN=2048


SAVE_PATH="/workspace/SVD-MOE-new/svd_share_mixtral"

# SAVE_PATH="/workspace/SVD-MOE-new"
DATASETS=("wikitext2")

# Expert Drop 参数
NUM_NODES=1
NUM_PROCESSES=1
DATASET="c4_train"
DATA_TYPE="pt"
N_COMPRESSION_SAMPLES=128
COMPRESS_METHOD="expert_drop"
# hkust
OUTPUT_DIR="/workspace/SVD-MOE-new/svd_share_mixtral/results"
# OUTPUT_DIR="/workspace/SVD-MOE-new/results/expert_drop"
REVERSE_DROP="False"
PRESERVE_GATE="False"

# 设置日志目录
COMBO_LOG_DIR="combo_log"
# 日志设置
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${COMBO_LOG_DIR}/combined_processing_log_${TIMESTAMP}.txt"
# 创建日志目录（如果不存在）
mkdir -p "$COMBO_LOG_DIR"
CONFIG_PATHS=(
  'keep_config_top_n-6.json'
  # 在这里添加更多的配置文件路径
)

# 日志函数
log_message() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") - $1" | tee -a "$LOG_FILE"
}

# 实验组设置
read -r -d '' EXPERIMENT_GROUPS << EOM
20%|3,5,7,9,12,15,20,23,26|NONE|3,5,7,9,12,15,20,23,26|0.2
EOM
# 20%|11,12,15,20,21,23,24,25|NONE|11,12,15,20,21,23,24,25|0.5
# 30%|11,12,15,16,20,21,23,24,25,27|NONE|11,12,15,16,20,21,23,24,25,27|0.7
# phiemoe
# 20%|11,12,15,20,21,23,24,25|NONE|11,12,15,20,21,23,24,25|0.5
# 30%|11,12,15,16,20,21,23,24,25,27|NONE|11,12,15,16,20,21,23,24,25,27|0.7
# 40%|6,9,11,12,13,15,16,18,20,21,23,24,25,27|NONE|6,9,11,12,13,15,16,18,20,21,23,24,25,27|0.8
# 50%|5,6,8,9,11,12,13,15,16,18,20,21,23,24,25,27,28,30|NONE|6,8,9,11,12,13,15,16,18,20,21,23,24,25,27,28,30|0.7
# 60%|5,6,8,9,11,12,13,15,16,18,20,21,23,24,25,27,28,30|NONE|6,8,9,11,12,13,15,16,18,20,21,23,24,25,27,28,30|0.7

# mixral
# 20%|3,5,7,9,10,12,23,24,26|NONE|3,5,7,9,10,12,23,24,26|0.2
# 1_50%|3,5,6,7,9,10,12,13,15,17,19,20,21,23,24,26,28,30|NONE|3,5,6,7,9,10,12,13,15,17,19,20,21,23,24,26,28,30|0.5
# 2_50%|3,5,6,7,9,10,12,13,15,17,19,20,21,23,24,26,28,30|4,8,11,14,16,18,22,25,27|3,5,6,7,9,10,12,13,15,17,19,20,21,23,24,26,28,30|0.4
# 1_60%|2,3,5,6,7,9,10,12,13,15,16,17,19,20,21,22,24,25,27,29,31|NONE|2,3,5,6,7,9,10,12,13,15,16,17,19,20,21,22,24,25,27,29,31|0.7
# 2_60%|2,3,5,6,7,9,10,12,13,15,16,17,19,20,21,22,24,25,27,29,31|4,8,11,14,18,23,26,28|2,3,5,6,7,9,10,12,13,15,16,17,19,20,21,22,24,25,27,29,31|0.5


# 1_50%|3,5,6,7,9,10,12,13,15,17,19,20,21,23,24,26,28,30|NONE|3,5,6,7,9,10,12,13,15,17,19,20,21,23,24,26,28,30|0.5
# 2_50%|3,5,6,7,9,10,12,13,15,17,19,20,21,23,24,26,28,30|4,8,11,14,16,18,22,25,27|3,5,6,7,9,10,12,13,15,17,19,20,21,23,24,26,28,30|0.4
# 1_60%|2,3,5,6,7,9,10,12,13,15,16,17,19,20,21,22,24,25,27,29,31|NONE|2,3,5,6,7,9,10,12,13,15,16,17,19,20,21,22,24,25,27,29,31|0.7
# 2_60%|2,3,5,6,7,9,10,12,13,15,16,17,19,20,21,22,24,25,27,29,31|4,8,11,14,18,23,26,28|2,3,5,6,7,9,10,12,13,15,16,17,19,20,21,22,24,25,27,29,31|0.5


# 3_5_7_9_10_12_23_24_26
# 30%|3,5,7,9,10,12,15,20,21,23,24,26|NONE|3,5,7,9,10,12,15,20,21,23,24,26|0.2
# 20%|3,5,7,9,10,12,23,24,26|NONE|3,5,7,9,10,12,23,24,26|0.2

# group_A|3,5,7,9,12,15,20,23,26|NONE|3,5,7,9,12,15,20,23,26|0.2

# phimoe

# 8层 20%
# 20%|11,12,15,20,21,23,24,25|NONE|11,12,15,20,21,23,24,25|0.5

# 10层，12层 30%
# 30%|11,12,15,16,20,21,23,24,25,27|NONE|11,12,15,16,20,21,23,24,25,27|0.7
# 30%|6,9,11,12,15,16,18,20,21,23,24,25|NONE|6,9,11,12,15,16,18,20,21,23,24,25|0.5

# 14层，16层 40%
# 40%|6,9,11,12,13,15,16,18,20,21,23,24,25,27|NONE|6,9,11,12,13,15,16,18,20,21,23,24,25,27|0.8
# 40%|6,8,9,11,12,13,15,16,18,20,21,23,24,25,27,28|NONE|6,8,9,11,12,13,15,16,18,20,21,23,24,25,27,28|0.6

# 0.3 18 50%
# 50%|5,6,8,9,11,12,13,15,16,18,20,21,23,24,25,27,28,30|NONE|6,8,9,11,12,13,15,16,18,20,21,23,24,25,27,28,30|0.7

# 0.3 22层 60%
# 60%|5,6,8,9,11,12,13,15,16,18,20,21,23,24,25,27,28,30|NONE|6,8,9,11,12,13,15,16,18,20,21,23,24,25,27,28,30|0.7


# SVD 处理
run_svd() {
    local SELECTED_LAYERS=$1
    local ATTENTION_LAYERS=$2
    local EXPERT_LAYERS=$3
    local RATIO=$4
    local EXP_GROUP=$5
    local STEP=$6

    log_message "Running SVD for group $EXP_GROUP with Selected Layers: $SELECTED_LAYERS, Attention Layers: $ATTENTION_LAYERS, Expert Layers: $EXPERT_LAYERS, Ratio: $RATIO, Step: $STEP"

    SVD_MODEL_PATH=""

    for DATASET in "${DATASETS[@]}"; do
        # 准备 Python 命令
        PYTHON_CMD="python SVDLLM_share_new.py \
            --model \"$MODEL_PATH\" \
            --step \"$STEP\" \
            --ratio \"$RATIO\" \
            --whitening_nsamples \"$WHITENING_NSAMPLES\" \
            --updating_nsamples \"$UPDATING_NSAMPLES\" \
            --dataset \"$DATASET\" \
            --seed \"$SEED\" \
            --model_seq_len \"$MODEL_SEQ_LEN\" \
            --save_path \"$SAVE_PATH\" \
            --DEV \"cuda\" \
            --eval_batch_size 1 \
            --group \"$EXP_GROUP\" \
            --evaluate_after_compression \
            --profiling_mat_path \"/workspace/SVD-MOE-new/svd_share_mixtral/3_5_7_9_12_15_20_23_26bothmistralai_Mixtral_8x7B_v0.1_profiling_wikitext2_1024_42.pt\""

        # 如果 selected_layers 不为 NONE，则添加到命令中
        if [ "$SELECTED_LAYERS" != "NONE" ] && [ -n "$SELECTED_LAYERS" ]; then
            PYTHON_CMD+=" --selected_layers \"$SELECTED_LAYERS\""
        fi

        # 如果 expert_layers 不为 NONE，则添加到命令中
        if [ "$EXPERT_LAYERS" != "NONE" ] && [ -n "$EXPERT_LAYERS" ]; then
            PYTHON_CMD+=" --expert_layers \"$EXPERT_LAYERS\""
        fi

        # 如果 attention_layers 不为 NONE，则添加到命令中
        if [ "$ATTENTION_LAYERS" != "NONE" ] && [ -n "$ATTENTION_LAYERS" ]; then
            PYTHON_CMD+=" --attention_layers \"$ATTENTION_LAYERS\""
        fi

        # 运行 SVD 压缩和评估，实时显示输出并记录到日志
        log_message "Executing command: $PYTHON_CMD"
        eval $PYTHON_CMD 2>&1 | tee -a "$LOG_FILE"

        # 从日志文件中提取 SVD 模型路径
        TEMP_PATH=$(grep -i "Saved model:" "$LOG_FILE" | tail -n 1 | awk '{print $NF}')
        
        if [ -n "$TEMP_PATH" ]; then
            SVD_MODEL_PATH=$TEMP_PATH
            log_message "Found SVD model path: $SVD_MODEL_PATH"
        fi
    done

    if [ -n "$SVD_MODEL_PATH" ]; then
        if [ -f "$SVD_MODEL_PATH" ]; then
            log_message "SVD model file confirmed at: $SVD_MODEL_PATH"
        else
            log_message "Error: SVD model file not found at path: $SVD_MODEL_PATH"
            SVD_MODEL_PATH=""
        fi
    else
        log_message "Error: Could not find SVD model path in output."
    fi
}

run_expert_drop() {
    local MODEL_PATH=$1
    local MODEL_NAME=$(basename "${MODEL_PATH}" .pt)
    local GROUP=$2

    log_message "Running Expert Drop for model: $MODEL_NAME"

    for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
        log_message "Processing config: $(basename "$CONFIG_PATH")"

        local OUTPUT_DIR="${OUTPUT_DIR}/${GROUP}/${MODEL_NAME}"
        local COMPRESSED_MODEL_SAVE_PATH="${OUTPUT_DIR}/checkpoint"
        echo "DEBUG: OUTPUT_DIR = ${OUTPUT_DIR}"
        echo "DEBUG: COMPRESSED_MODEL_SAVE_PATH = ${COMPRESSED_MODEL_SAVE_PATH}"
        # 构建 Expert Drop 命令，确保所有参数都正确引用
        EXPERT_DROP_CMD="accelerate launch \
            --config_file \"config/accelerate/mixtral_normal.yaml\" \
            --num_processes ${NUM_PROCESSES} \
            --num_machines ${NUM_NODES} \
            src/run_prune.py \
            --stage prune \
            --model_name_or_path \"${MODEL_PATH}\" \
            --dataset \"${DATASET}\" \
            --split \"train\" \
            --data_type \"${DATA_TYPE}\" \
            --cutoff_len ${MODEL_SEQ_LEN} \
            --output_dir \"${OUTPUT_DIR}\" \
            --logging_steps 10 \
            --bf16 \
            --n_compression_samples ${N_COMPRESSION_SAMPLES} \
            --compress_method ${COMPRESS_METHOD} \
            --expert_drop_method \"post_dropping\" \
            --reverse_drop ${REVERSE_DROP} \
            --preserve_gate ${PRESERVE_GATE} \
            --compressed_model_save_path \"${COMPRESSED_MODEL_SAVE_PATH}\" \
            --config_path \"${CONFIG_PATH}\""

        log_message "Executing Expert Drop command: $EXPERT_DROP_CMD"
        eval $EXPERT_DROP_CMD 2>&1 | tee -a "$LOG_FILE"

        if [ $? -ne 0 ]; then
            log_message "Error: Expert Drop failed for model $MODEL_NAME"
            return 1
        fi

        log_message "Expert Drop processing completed for config: $(basename "$CONFIG_PATH")"

        # Run the evaluation script
        log_message "Executing evaluation command..."
        EVAL_CMD="python results/evluate_ppl.py \
            --compressed_model_save_path \"${COMPRESSED_MODEL_SAVE_PATH}\" \
            --config_path \"${CONFIG_PATH}\" \
            --model_name \"${MODEL_NAME}\""

        eval $EVAL_CMD 2>&1 | tee -a "$LOG_FILE"

        if [ $? -ne 0 ]; then
            log_message "Error: Evaluation failed for model $MODEL_NAME"
            log_message "Evaluation command was: $EVAL_CMD"
            return 1
        fi

        log_message "PPL Evaluation completed for model: $MODEL_NAME"

        # Run the lm-evaluation-harness
        local TUNE_ID="${COMPRESSED_MODEL_SAVE_PATH}/${MODEL_NAME}${CONFIG_PATH/.json/}.pt"
        log_message "Running lm-evaluation-harness for tune_id: $TUNE_ID"

        LM_EVAL_CMD="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python lm-evaluation-harness/lm_eval --model ${TUNE_ID} \
            --model_args parallelize=True \
            --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,mathqa \
            --output_path \"${COMPRESSED_MODEL_SAVE_PATH}\" \
            --batch_size 1"

        eval $LM_EVAL_CMD 2>&1 | tee -a "$LOG_FILE"

        if [ $? -ne 0 ]; then
            log_message "Error: lm-evaluation-harness failed for model $MODEL_NAME"
            log_message "lm-evaluation-harness command was: $LM_EVAL_CMD"
            return 1
        fi

        log_message "lm-evaluation-harness completed for model: $MODEL_NAME"

        # 删除 TUNE_ID 文件
        if [ "$DELETE_TUNE_ID" == "True" ]; then
            if [ -f "$TUNE_ID" ]; then
                log_message "Attempting to delete TUNE_ID file: $TUNE_ID"
                if rm "$TUNE_ID"; then
                    log_message "Successfully deleted TUNE_ID file: $TUNE_ID"
                else
                    log_message "Error: Failed to delete TUNE_ID file: $TUNE_ID"
                fi
            else
                log_message "Warning: TUNE_ID file not found: $TUNE_ID"
            fi
        else
            log_message "Skipping deletion of TUNE_ID file: $TUNE_ID"
        fi
    done

    return 0
}

# 主处理流程
main() {
    local STEP=1
    DELETE_TUNE_ID="False"  # 设置开关，控制是否删除 $TUNE_ID
    
    while IFS='|' read -r GROUP SELECTED_LAYERS ATTENTION_LAYERS EXPERT_LAYERS RATIO; do
        log_message "Processing Experiment Group: $GROUP"
        
        log_message "Parsed values:"
        log_message "SELECTED_LAYERS: $SELECTED_LAYERS"
        log_message "ATTENTION_LAYERS: $ATTENTION_LAYERS"
        log_message "EXPERT_LAYERS: $EXPERT_LAYERS"
        log_message "RATIO: $RATIO"
        
        # 运行 SVD 处理
        run_svd "$SELECTED_LAYERS" "$ATTENTION_LAYERS" "$EXPERT_LAYERS" "$RATIO" "$GROUP" "$STEP"
        # SVD_MODEL_PATH="/workspace/SVD-MOE-new/svd_share_mixtral/attnexpert3_5_7_9_10_12_23_24_26mistralai_Mixtral_8x7B_v0.1_whitening_only_0.8_20%.pt"
        log_message "After run_svd, SVD_MODEL_PATH is: $SVD_MODEL_PATH"
        
        if [ -n "$SVD_MODEL_PATH" ] && [ -f "$SVD_MODEL_PATH" ]; then
            log_message "SVD model file found: $SVD_MODEL_PATH"
            if run_expert_drop "$SVD_MODEL_PATH" "$GROUP"; then
                log_message "Expert Drop completed successfully"
                
                # 删除 SVD 模型文件
                # if rm "$SVD_MODEL_PATH"; then
                #     log_message "SVD model file deleted successfully: $SVD_MODEL_PATH"
                # else
                #     log_message "Error: Failed to delete SVD model file: $SVD_MODEL_PATH"
                # fi

            else
                log_message "Expert Drop failed, skipping evaluation"
            fi
        else
            log_message "Error: SVD model file not found or path is empty. Skipping Expert Drop for this configuration."
        fi
        
    done <<< "$EXPERIMENT_GROUPS"
}

# 运行主处理流程
main