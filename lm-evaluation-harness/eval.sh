export PYTHONPATH='./lm-evaluation-harness':'./lm-evaluation-harness'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python lm-evaluation-harness/lm_eval --model ${TUNE_ID} \
            --model_args parallelize=True \
            --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,mathqa \
            --output_path \"${COMPRESSED_MODEL_SAVE_PATH}\" \
            --batch_size 1