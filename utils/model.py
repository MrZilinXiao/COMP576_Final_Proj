import torch
import gc
import time
from loguru import logger
from collections import defaultdict
import torch.nn as nn


def create_act(act, num_parameters=None):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'prelu':
        return nn.PReLU(num_parameters)
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'linear':
        class Identity(nn.Module):
            def forward(self, x):
                return x

        return Identity()
    else:
        raise ValueError('Unknown activation function {}'.format(act))


def model_param_counter(model: torch.nn.Module, split_list=None):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    res = {
        'total': total_params,
        'trainable': total_trainable_params,
    }
    if split_list is not None:
        split_count = defaultdict(int)
        assert isinstance(split_list, list)
        for key_word in split_list:
            for kv in model.named_parameters():  # kv[0], kv[1]
                if key_word in kv[0]:
                    split_count[key_word] += kv[1].numel()

        # merge res and split_count
        res.update(split_count)

    return res


def collect_mem():
    """
    collect gpu & cpu memory after each training epoch
    :return:
    """
    torch.cuda.empty_cache()
    gc.collect()


def load_state_dicts(model, checkpoint_file, map_location=None, ignore_keys=None, specific_state_key=None):
    """Load torch items from saved state_dictionaries.
    """
    if map_location is None:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
    else:
        checkpoint = torch.load(checkpoint_file, map_location=map_location)

    if specific_state_key is not None:
        checkpoint = checkpoint[specific_state_key]

    elif 'state_dict' in checkpoint:  # only use `state_dict`
        checkpoint = checkpoint['state_dict']

    # filter `ignore_keys` in checkpoint
    load_model_keys = checkpoint.keys()
    if ignore_keys is not None:
        for ignore_key in ignore_keys:
            load_model_keys = [k for k in load_model_keys if ignore_key not in k]

    checkpoint = {k.replace('module.', ''): checkpoint[k] for k in load_model_keys}
    # WARNING: makes sure `module.` is not included in normal parameters

    load_status = model.load_state_dict(checkpoint, strict=False)
    if load_status is not None and str(load_status) != '<All keys matched successfully>':
        logger.warning("Caught some errors when loading state_dict, \n" +
                       f"missing keys: {load_status.missing_keys}\nunexpected_keys: {load_status.unexpected_keys}")

    if load_status is not None and str(load_status) == '<All keys matched successfully>':
        logger.info(f"Loaded weight from {checkpoint_file}! No parameters mismatch!")
