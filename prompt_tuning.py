"""
Example usage with 1 GPU: 
CUDA_VISIBLE_DEVICES=1 python prompt_tuning.py --exp-name MELD_roberta_large_prompt --dataset MELD --dataset-root ../data --init-lr 3e-6 --max-train-epochs 5 --batch-size 4 --save-args --experiment-tag roberta-large
"""
import os.path as osp
import os

from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptForClassification, PromptDataLoader
from openprompt.data_utils import InputExample

from utils.meld_dataset import MELDDataset
import torch
import torch.nn as nn
from transformers.trainer_pt_utils import get_parameter_names

from tqdm import tqdm
from utils.my_metrics import general_compute_metrics
from sklearn import metrics

from utils.my_metrics import AverageMeter
from termcolor import colored
from utils.model import model_param_counter, collect_mem
from utils.general import str2bool, create_logger

import argparse
from datetime import datetime
import wandb

from utils.general import create_logger, create_dir
import pprint
import json

classes = ['neutral',
           'surprise',
           'fear',
           'sadness',
           'joy',
           'disgust',
           'anger']

emotion_to_idx = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 
                  'disgust': 5, 'anger': 6}

BEST_WEIGHTED_F1 = float('-inf')

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
    parser.add_argument('--warmup-ratio', type=float, default=0.2, help='Linear warmup ratio')
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
    
    # add for prompt tuning customized args
    parser.add_argument('--print-freq', type=int, default=5)
    parser.add_argument('--val-freq', type=int, default=1)
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

def get_prompt_dataset(args):
    train_ds = MELDDataset(data_root=args.dataset_root, tokenizer=None, 
                           split='train', prompt_dataset=True)
    val_ds = MELDDataset(data_root=args.dataset_root, tokenizer=None,
                            split='dev', prompt_dataset=True)
    test_ds = MELDDataset(data_root=args.dataset_root, tokenizer=None,
                            split='test', prompt_dataset=True)
    
    real_train_ds = [
        InputExample(guid=idx, text_a=uttr, label=label)
        for idx, (uttr, label) in enumerate(train_ds)
    ]
    real_val_ds = [
        InputExample(guid=idx, text_a=uttr, label=label)
        for idx, (uttr, label) in enumerate(val_ds)
    ]
    real_test_ds = [
        InputExample(guid=idx, text_a=uttr, label=label)
        for idx, (uttr, label) in enumerate(test_ds)
    ]
    
    return real_train_ds, real_val_ds, real_test_ds

@torch.no_grad()
def predict_id_label(logits):
    return torch.argmax(logits, dim=-1)

def train(model: PromptForClassification, train_loader, epoch):
    model.train()
    model.zero_grad()
    print(('\n' + '%14s' * 5) % ('epoch', 'gpu_mem', 'LR', 'loss', 'acc'))
    average_meters = {
        key: AverageMeter(key) for key in ['acc', 'loss']
    }
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    gt_list, pred_list = [], []

    for i, batch in pbar:
        optimizer.zero_grad()
        
        for k in batch.keys():
            batch[k] = batch[k].to(device)

        logits = model(batch)
        loss = criterion(logits, batch['label'])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        # predict during training...
        label = batch['label'].cpu().tolist()
        pred = predict_id_label(logits).cpu().tolist()
        gt_list.extend(label)
        pred_list.extend(pred)

        acc = metrics.accuracy_score(label, pred)
        # record during training an epoch
        average_meters['loss'].update(loss.item(), batch['label'].shape[0])
        average_meters['acc'].update(acc, batch['label'].shape[0])

        if (i + 1) % args.print_freq == 0 or (i + 1) == len(train_loader):

            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%14s' * 2 + '%14.7g' + '%14.5g' * 2) % ('%d/%d' % (epoch + 1, args.max_train_epochs),
                                                          mem,
                                                          optimizer.state_dict()['param_groups'][0]['lr'],
                                                          average_meters['loss'].avg,
                                                          average_meters['acc'].avg
                                                          )

            pbar.set_description(s)

    epoch_metrics = general_compute_metrics(gt_list, pred_list, labels=[0, 1, 2, 3, 4, 5, 6], text_labels=classes)

    logger.info(
        colored("Train Epoch {}/{}: loss:{:.6f}, acc:{:.6f}, weighted-f1: {:.6f}, micro-f1: {:.6f}".format(epoch + 1,
                                                                                                           args.max_train_epochs,
                                                                                                           average_meters[
                                                                                                               'loss'].avg,
                                                                                                           metrics.accuracy_score(
                                                                                                               gt_list,
                                                                                                               pred_list),
                                                                                                           epoch_metrics[
                                                                                                               'weighted_f1'],
                                                                                                           epoch_metrics[
                                                                                                               'f1_micro']
                                                                                                           ),
                color='green')
    )
    logger.debug(epoch_metrics['class_report'])
    print(epoch_metrics['class_report'])
    # wandb log
    wandb.log({'train/loss': average_meters['loss'].avg, 'train/acc': metrics.accuracy_score(gt_list, pred_list),
               'train/f1_weighted': epoch_metrics['weighted_f1']})

@torch.no_grad()
def val(model: PromptForClassification, val_loader, epoch, split='val'):
    global BEST_WEIGHTED_F1
    model.eval()
    # print(('\n' + '%14s' * 5) % ('epoch', 'gpu_mem', 'LR', 'loss', 'acc'))
    average_meters = {
        key: AverageMeter(key) for key in ['acc', 'loss']
    }
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    gt_list, pred_list = [], []

    for i, batch in pbar:
        for k in batch.keys():
            batch[k] = batch[k].to(device)

        # with autocast():
        logits = model(batch)
        loss = criterion(logits, batch['label'])

        # predict during training...
        label = batch['label'].cpu().tolist()
        pred = predict_id_label(logits).cpu().tolist()
        gt_list.extend(label)
        pred_list.extend(pred)

        average_meters['loss'].update(loss.item(), batch['label'].shape[0])

    epoch_metrics = general_compute_metrics(gt_list, pred_list, labels=[0, 1, 2, 3, 4, 5, 6], text_labels=classes)
    logger.info(
        "{} Epoch {}/{}: loss:{:.6f}, acc:{:.6f}, weighted-f1: {:.6f}, micro-f1: {:.6f}, macro-f1: {:.6f}".format(
            split,
            epoch + 1,
            args.max_train_epochs,
            average_meters[
                'loss'].avg,
            metrics.accuracy_score(
                gt_list,
                pred_list),
            epoch_metrics['weighted_f1'],
            epoch_metrics['f1_micro'], epoch_metrics['f1_macro']))
    # should record other rec, pre, class-f1!
    logger.debug(epoch_metrics['class_report'])
    print(epoch_metrics['class_report'])
    if split == 'val':
        wandb.log({'eval/loss': average_meters['loss'].avg, 'eval/acc': metrics.accuracy_score(gt_list, pred_list),
                   'eval/f1_weighted': epoch_metrics['weighted_f1']}, step=epoch + 1)
        # don't record eval best metric
    elif split == 'test':
        wandb.log({'test/loss': average_meters['loss'].avg, 'test/acc': metrics.accuracy_score(gt_list, pred_list),
                   'test/f1_weighted': epoch_metrics['weighted_f1']}, step=epoch + 1)
        if epoch_metrics['weighted_f1'] > BEST_WEIGHTED_F1:
            BEST_WEIGHTED_F1 = epoch_metrics['weighted_f1']
            wandb.log({'test/best_f1_weighted': BEST_WEIGHTED_F1, 'test/best_epoch': epoch + 1}, step=epoch + 1)
            torch.save(model.state_dict(), os.path.join(args.log_dir,
                                                        'best.pt'))

if __name__ == '__main__':
    args = parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if args.experiment_tag:
        args.log_dir = os.path.join(args.log_dir, args.experiment_tag)
    else:
        args.log_dir = os.path.join('prompt_log', timestamp)
    wandb.init(config=args)
    os.makedirs(args.log_dir)
    logger = create_logger(args.log_dir)
    
    device = torch.device('cuda:' + str(args.device))
    
    logger.info('loading pretrained PLM...')
    model_name = args.model.split('-')[0]
    plm, tokenizer, model_config, WrapperClass = load_plm(model_name, args.model)
    logger.info('loading prompt utils...')
    promptTemplate = ManualTemplate(
        text='{"placeholder":"text_a"} The speaker is in the {"mask"} emotion.',
        tokenizer=tokenizer,
    )
    promptVerbalizer = ManualVerbalizer(
        classes=classes,
        label_words={
            _class: [_class] for _class in classes
        },
        tokenizer=tokenizer,
    )
    
    promptModel = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=promptVerbalizer,
    )
    logger.info(f'model params info: {model_param_counter(promptModel)}')

    logger.info('building prompt dataset...')
    train_ds, val_ds, test_ds = get_prompt_dataset(args)
    
    logger.info('building prompt dataloader...')
    
    train_loader = PromptDataLoader(
        dataset=train_ds,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size=args.batch_size
    )

    val_loader = PromptDataLoader(
        dataset=val_ds,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size=args.batch_size * 2,
        shuffle=False
    )

    test_loader = PromptDataLoader(
        dataset=test_ds,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size=args.batch_size * 2,
        shuffle=False
    )
    
    promptModel = promptModel.to(device)
    decay_parameters = get_parameter_names(promptModel, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in promptModel.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in promptModel.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.init_lr)  # fix default weight_decay
    # begin train
    for epoch in range(args.max_train_epochs):
        train(promptModel, train_loader, epoch)
        # deal with OOM
        collect_mem()
        val(promptModel, val_loader, epoch, split='val')
        val(promptModel, test_loader, epoch, split='test')