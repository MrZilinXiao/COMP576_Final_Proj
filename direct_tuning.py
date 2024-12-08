"""
Example usage with 1 GPU: 
CUDA_VISIBLE_DEVICES=0 python direct_tuning.py --exp-name MELD_roberta_large --dataset MELD --dataset-root ../data --init-lr 3e-6 --max-train-epochs 5 --batch-size 4 --log-dir ./logs --save-args --experiment-tag direct_tuning_nowarm
"""
import os
import os.path as osp
import json
import argparse
import pprint
from datetime import datetime
from utils.general import create_logger, create_dir
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer)
from utils.my_metrics import compute_metrics
from utils.model import model_param_counter
from utils.meld_dataset import MELDDataset

def parse_args():
    parser = argparse.ArgumentParser(description='MultiERC Arguments')
    #
    # self-defined params for exp ease
    #
    parser.add_argument('--exp-name', type=str, help='Experiment name', default='default')
    parser.add_argument('--experiment-tag', type=str, default=None, help='will be used to name a subdir '
                                                                         'for log-dir if given')
    parser.add_argument('--note', type=str, help='Experiment note', default='default_note')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--skip-train', action='store_true', default=False)  # do not training, just val
    parser.add_argument('--log-dir', type=str, default='logs/', help='where to save training-progress, model, etc')
    #
    # Dataset-oriented arguments
    #
    parser.add_argument('--dataset', type=str, default='MELD', choices=['MELD'])
    parser.add_argument('--dataset-root', type=str, default='./data')
    #
    # Training arguments
    #
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'])
    parser.add_argument('--max-train-epochs', type=int, default=5, help='number of training epochs.')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--init-lr', type=float, default=3e-6, help='initial learning rate for training.')
    parser.add_argument('--weight-decay', type=float, default=0, help='weight decay of optimizer')
    parser.add_argument('--warmup-ratio', type=float, default=0, help='Linear warmup ratio')
    #
    # Model arguments
    #
    parser.add_argument('--model', type=str, default='roberta-large')
    parser.add_argument('--no-pretrain', action='store_true', default=False)
    #
    # Misc arguments
    #
    parser.add_argument('--device', type=str, default='0', help='specify gpu device. [default: 0]')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size per gpu. [default: 4]')
    parser.add_argument('--save-args', action='store_true', default=True, help='save arguments in a json.txt')
    # Parallel Arguments, Do not Modify
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP Parameter, do not modify!')
    parser.add_argument('--deepspeed', type=str, default=None)
    
    args = parser.parse_args()
    
    # post processing
    if args.log_dir:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # if args.pretrain:
        #     args.log_dir = args.log_dir
        if args.experiment_tag:
            args.log_dir = osp.join(args.log_dir, args.experiment_tag, timestamp)
        else:
            args.log_dir = osp.join(args.log_dir, timestamp)

        args.checkpoint_dir = create_dir(osp.join(args.log_dir, 'checkpoints'), args.local_rank)
        args.tensorboard_dir = create_dir(osp.join(args.log_dir, 'tb_logs'), args.local_rank)
        
    # Print them nicely
    if args.local_rank in [-1, 0]:
        args_string = pprint.pformat(vars(args))
        print(args_string)

    if args.save_args and args.local_rank in [-1, 0]:
        out = osp.join(args.log_dir, 'config.json.txt')
        with open(out, 'w') as f_out:
            json.dump(vars(args), f_out, indent=4, sort_keys=True)

    return args

args = parse_args()
logger = create_logger(args.log_dir, local_rank=args.local_rank)
logger.info('inited logger!')

def main():
    DATASET = args.dataset
    ROOT_DIR = args.dataset_root
    SEED = args.random_seed
    NUM_CLASSES = 7
    
    BATCH_SIZE = args.batch_size
    OUTPUT_DIR = args.log_dir

    EVALUATION_STRATEGY = 'epoch'
    LOGGING_STRATEGY = 'epoch'
    SAVE_STRATEGY = 'no'

    PER_DEVICE_TRAIN_BATCH_SIZE = BATCH_SIZE
    PER_DEVICE_EVAL_BATCH_SIZE = BATCH_SIZE * 2

    LOAD_BEST_MODEL_AT_END = False

    METRIC_FOR_BEST_MODEL = 'eval_f1_weighted'
    GREATER_IS_BETTER = True

    WARMUP_RATIO = args.warmup_ratio
    WEIGHT_DECAY = args.weight_decay

    train_args = dict(output_dir=OUTPUT_DIR,
                      evaluation_strategy=EVALUATION_STRATEGY,
                      logging_strategy=LOGGING_STRATEGY,
                      save_strategy=SAVE_STRATEGY,
                      per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
                      per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
                      load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
                      seed=SEED,
                      learning_rate=args.init_lr,
                      num_train_epochs=args.max_train_epochs,
                      weight_decay=WEIGHT_DECAY,
                      warmup_ratio=WARMUP_RATIO,
                      metric_for_best_model=METRIC_FOR_BEST_MODEL,
                      greater_is_better=GREATER_IS_BETTER,
                      dataloader_num_workers=args.num_workers,
                      local_rank=args.local_rank,
                      ddp_find_unused_parameters=False,
                      remove_unused_columns=True, 
                      run_name=args.exp_name,
                      )

    if args.deepspeed is not None:
        train_args['deepspeed'] = args.deepspeed
        
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=NUM_CLASSES)
    logger.info(f'init {args.model} with param info: {model_param_counter(model)}!')

    train_args = TrainingArguments(**train_args)
    train_ds = MELDDataset(data_root=args.dataset_root, tokenizer=tokenizer, 
                           split='train', prompt_dataset=False)
    val_ds = MELDDataset(data_root=args.dataset_root, tokenizer=tokenizer,
                            split='dev', prompt_dataset=False)
    test_ds = MELDDataset(data_root=args.dataset_root, tokenizer=tokenizer,
                            split='test', prompt_dataset=False)

    logger.info('init trainer...')

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # print(f"trainer data_collator: {trainer.data_collator}")

    if not args.skip_train:
        logger.info('training...')
        trainer.train()

    logger.info(f"eval ...")
    val_results = trainer.evaluate()

    with open(os.path.join(OUTPUT_DIR, 'val-results.json'), 'w') as stream:
        json.dump(val_results, stream, indent=4)
    logger.info(f"eval results: {val_results}")
    
    logger.info(f"test ...")
    test_results = trainer.predict(test_ds)
    with open(os.path.join(OUTPUT_DIR, 'test-results.json'), 'w') as stream:
        json.dump(test_results.metrics, stream, indent=4)
    logger.info(f"test results: {test_results.metrics}")


if __name__ == '__main__':
    main()
