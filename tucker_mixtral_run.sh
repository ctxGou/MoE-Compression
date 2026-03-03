#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false


# 限制 CPU 线程数
export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=18
export NUMEXPR_NUM_THREADS=18
export OPENBLAS_NUM_THREADS=18
export HF_DATASETS_TRUST_REMOTE_CODE=1

# 模型和路径配置
SAVE_PATH="./results"
DATASET="wikitext2"
CLUSTER_TYPE="global" # global group mixed 
# important_layers=(10 13 21 22 26)

############# Mixtral
MODEL_PATH="mistralai/Mixtral-8x7B-v0.1"

RATIO=0.2 # mixtral
LAYERS_TO_COMPRESS=(3 5 6 7 9 12 23 24 25) # mixtral 0.2 global
# LAYERS_TO_COMPRESS=(3 5 6) # mixtral 0.2 global
# LAYERS_TO_COMPRESS=(7 9 12) # mixtral 0.2 global
# LAYERS_TO_COMPRESS=(23 24 25) # mixtral 0.2 global

# RATIO=0.4 
# LAYERS_TO_COMPRESS=(3 5 6 7 9 10 12 13 21 22 23 24 25 26) # mixtral 0.4
# LAYERS_TO_COMPRESS=(3 5 6 7 9) # mixtral 0.4
# LAYERS_TO_COMPRESS=(10 12 13 21 22) # mixtral 0.4
# LAYERS_TO_COMPRESS=(23 24 25 26) # mixtral 0.4


# LAYERS_TO_COMPRESS=(12 13 21 22 ) # mixtral 0.4
# LAYERS_TO_COMPRESS=(23 24 25 26) # mixtral 0.4


# RATIO=0.6
# LAYERS_TO_COMPRESS=(2 3 5 6 7 9 10 12 13 14 15 16 17 20 21 22 23 24 25 26 27) # mixtral 0.6



# python src/run_tucker.py \
#     --model_path $MODEL_PATH \
#     --save_path $SAVE_PATH \
#     --whitening_nsamples 256 \
#     --cluster_type "global" \
#     --model_seq_len 2048 \
#     --whiten_type "input" \
#     --layers_to_compress ${LAYERS_TO_COMPRESS[@]} \
#     --ratio $RATIO \
#     --decomposition_method "svd" \
#     --run_eval True \
#     --ppl_datasets wikitext2 ptb c4 \
#     --eval_tasks openbookqa winogrande piqa arc_easy arc_challenge    \

# 计算协方差
# python src/run_covariances.py \
#     --model_path $MODEL_PATH \
#     --save_path $SAVE_PATH \
#     --dataset $DATASET \
#     --layers_to_compress ${LAYERS_TO_COMPRESS[@]} \
#     --whitening_nsamples 256 \
#     --cluster_type $CLUSTER_TYPE \
#     --num_clusters $NUM_CLUSTERS \
#     --model_seq_len 2048 \

# 计算白化矩阵
# python src/run_whitening.py \
#     --model_path $MODEL_PATH \
#     --save_path $SAVE_PATH \
#     --ratio $RATIO \
#     --layers_to_compress ${LAYERS_TO_COMPRESS[@]} \
#     --whitening_nsamples 256 \
#     --cluster_type "global" \
#     --model_seq_len 2048 \
#     --whiten_type "output" \




# 评估 记得换层
python src/run_evaluation.py \
    --model_path $MODEL_PATH \
    --save_path $SAVE_PATH \
    --dataset $DATASET \
    --ratio $RATIO \
    --layers_to_compress ${LAYERS_TO_COMPRESS[@]} \
    --cluster_type $CLUSTER_TYPE \
    --model_seq_len 2048 \
    --eval_batch_size 8 \
    --lm_eval_batch_size 96 \
    --whiten_type "none" \
    --eval_tasks piqa arc_easy arc_challenge mathqa hellaswag\
    --ppl_datasets wikitext2 ptb c4 \
    # --eval_tasks openbookqa winogrande piqa arc_easy arc_challenge mathqa hellaswag \



# nohup bash tucker_mixtral.sh > ./logs/global_mixtral_0_4_both_decomp_2.log 2>&1 &
# nohup bash tucker_mixtral.sh > ./logs/global_mixtral_0_4_both_eval.log 2>&1 & 



