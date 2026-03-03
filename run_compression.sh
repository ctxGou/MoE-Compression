#!/bin/bash

# 设置环境变量
# export CUDA_VISIBLE_DEVICES=4,5,6
export CUDA_VISIBLE_DEVICES=6
export TOKENIZERS_PARALLELISM=false


# 限制 CPU 线程数
export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=18
export NUMEXPR_NUM_THREADS=18
export OPENBLAS_NUM_THREADS=18


# 模型和路径配置
SAVE_PATH="./results"
DATASET="wikitext2"
CLUSTER_TYPE="global" # global group mixed 

# important_layers=(10 13 21 22 26)
# important_layers=(5 21 22 23 24 25 26)

############# mixtral
# MODEL_PATH="./models/Mixtral-8x7B-v0.1"

# RATIO=0.2 # mixtral
# LAYERS_TO_COMPRESS=(3 5 6 7 9 12 23 24 25) # mixtral 0.2 global
# LAYERS_TO_COMPRESS=(3 5 6) 
# LAYERS_TO_COMPRESS=(7 9 12)
# LAYERS_TO_COMPRESS=(23 24 25)


# RATIO=0.4 
# LAYERS_TO_COMPRESS=(3 5 6 7 9 10 12 13 21 22 23 24 25 26) # mixtral 0.4
# LAYERS_TO_COMPRESS=(3 5 6) # mixtral 0.4
# LAYERS_TO_COMPRESS=(7 9 10) # mixtral 0.4
# LAYERS_TO_COMPRESS=(12 13 21 22) # mixtral 0.4
# LAYERS_TO_COMPRESS=(23 24 25 26) # mixtral 0.4
# LAYERS_TO_COMPRESS=(3 5 6 7 9 10 12 13) # mixtral 0.4
# LAYERS_TO_COMPRESS=(21 22 23 24 25 26) # mixtral 0.4
# LAYERS_TO_COMPRESS=(22)


# RATIO=0.6
# LAYERS_TO_COMPRESS=(2 3 5 6 7 9 10 12 13 14 15 16 17 20 21 22 23 24 25 26 27) # mixtral 0.6

############# phi
# MODEL_PATH="./models/Phi-3.5-MoE-instruct"


# RATIO=0.2 # phi
# LAYERS_TO_COMPRESS=(11 12 15 20 21 23 24 25) # phi 0.2
# LAYERS_TO_COMPRESS=(11 12 15 20) # phi 0.2
# LAYERS_TO_COMPRESS=(21 23 24 25) # phi 0.2

# RATIO=0.4 # phi
# LAYERS_TO_COMPRESS=(10 11 12 15 16 18 20 21 23 24 25 26 27 28) # phi 0.4
# LAYERS_TO_COMPRESS=(10 11 12 15 16) # phi 0.4
# LAYERS_TO_COMPRESS=(18 20 21 23 24) # phi 0.4
# LAYERS_TO_COMPRESS=(25 26 27 28) # phi 0.4


# LAYERS_TO_COMPRESS=(18 20 21)
# LAYERS_TO_COMPRESS=(22 23 24)
# LAYERS_TO_COMPRESS=(25 26 27)


# 计算白化矩阵
# python src/run_whitening_ada.py \
#     --model_path $MODEL_PATH \
#     --save_path $SAVE_PATH \
#     --whitening_nsamples 256 \
#     --cluster_type "global" \
#     --model_seq_len 2048 \
#     --whiten_type "output" \
#     --ratio_alloc_csv_path \



# 评估 input/output
python src/run_evaluation.py \
    --model_path $MODEL_PATH \
    --save_path $SAVE_PATH \
    --dataset $DATASET \
    --ratio $RATIO \
    --layers_to_compress ${LAYERS_TO_COMPRESS[@]} \
    --cluster_type $CLUSTER_TYPE \
    --model_seq_len 2048 \
    --eval_batch_size 8 \
    --whiten_type "output" \
#     --ppl_datasets wikitext2 ptb \
#     --eval_tasks openbookqa winogrande arc_easy arc_challenge piqa   \
    # --important_layers ${important_layers[@]} \
    # --global_layers ${GLOBALLAYERS[@]} 


# nohup bash run_compression.sh > ./logs/global_mixtral_0_4_both_eval.log 2>&1 &
# nohup bash run_compression.sh > ./logs/global_mixtral_0_2_output.log 2>&1 &
# nohup bash run_compression.sh > ./logs/global_mixtral_0_6_decomp.log 2>&1 &
