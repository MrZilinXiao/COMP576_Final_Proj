# search over lr, weight_decay and batch_size on roberta_large
for lr in 3e-6 1e-5 3e-5 1e-6
do
    for weight_decay in 0 0.01 0.001 0.0001
    do
        for batch_size in 4 8 16
        do
            WANDB_PROJECT=2024Fall_576 CUDA_VISIBLE_DEVICES=7 python direct_tuning.py \
                --exp-name "MELD_roberta_large_lr${lr}_wd${weight_decay}_bs${batch_size}" \
                --dataset MELD \
                --dataset-root ../data \
                --init-lr $lr \
                --weight-decay $weight_decay \
                --max-train-epochs 5 \
                --batch-size $batch_size \
                --log-dir ./logs \
                --save-args \
                --model roberta-large \
                --experiment-tag roberta_large
        done
    done
done