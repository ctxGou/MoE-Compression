#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2
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
NUM_CLUSTERS=2
# important_layers=(10 13 21 22 26)

############# phi
MODEL_PATH="./models/Phi-3.5-MoE-instruct"
RATIO=0.2 # phi
LAYERS_TO_COMPRESS=(11 12 15 20 21 23 24 25)

# LAYERS_TO_COMPRESS=(11 12 15)
# LAYERS_TO_COMPRESS=(20 21 23)
# LAYERS_TO_COMPRESS=(24 25)

# RATIO=0.4 # phi
# LAYERS_TO_COMPRESS=(10 11 12 15 16 18 20 21 23 24 25 26 27 28) # phi 0.4

# RATIO=0.6 # phi
# LAYERS_TO_COMPRESS=(5 6 10 11 12 13 15 16 18 19 20 21 23 24 25 26 27 28 29) # phi 0.6



python src/run_tucker.py \
    --model_path $MODEL_PATH \
    --save_path $SAVE_PATH \
    --whitening_nsamples 256 \
    --cluster_type "global" \
    --model_seq_len 2048 \
    --whiten_type "output" \
    --layers_to_compress ${LAYERS_TO_COMPRESS[@]} \
    --ratio $RATIO \
    --run_eval True \
    --ppl_datasets wikitext2 ptb c4 \
    --eval_tasks openbookqa winogrande piqa arc_easy arc_challenge    \


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
#     --whiten_type "input" \

# python src/run_evaluation.py \
#     --model_path $MODEL_PATH \
#     --save_path $SAVE_PATH \
#     --dataset $DATASET \
#     --ratio $RATIO \
#     --layers_to_compress ${LAYERS_TO_COMPRESS[@]} \
#     --cluster_type $CLUSTER_TYPE \
#     --model_seq_len 2048 \
#     --eval_batch_size 8 \
#     --whiten_type "both" \
#     --ppl_datasets wikitext2 ptb c4 \
#     --eval_tasks openbookqa winogrande piqa arc_easy arc_challenge    \


# nohup bash tucker_phi.sh > ./logs/global_phi_0_4_input_decomp.log 2>&1 &
# nohup bash tucker_phi.sh > ./logs/global_phi_0_4_both_eval.log 2>&1 & 



