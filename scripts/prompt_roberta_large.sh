CUDA_VISIBLE_DEVICES=1 python prompt_tuning.py \
    --exp-name MELD_roberta_large_prompt \
    --dataset MELD \
    --dataset-root ../data \
    --init-lr 3e-6 \
    --max-train-epochs 5 \
    --batch-size 4 \
    --save-args \
    --model roberta-large \
    --experiment-tag roberta-large