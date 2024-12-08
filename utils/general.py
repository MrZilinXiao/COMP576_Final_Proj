import argparse
import random
import os.path as osp
from loguru import logger
import sys
import os
import numpy as np
import torch

def create_dir(dir_path, local_rank=-1):
    """
    Creates a directory (or nested directories) if they don't exist.
    """
    if local_rank not in [-1, 0]:
        return dir_path

    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path

def Singleton(cls):
    """
    A decorator for Singleton support
    """
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton


@Singleton
class DummyLogger(object):
    """
    Overwrite all logger methods in non-main workers
    """

    def __init__(self):
        pass

    def debug(self, msg, *args, **kwargs):
        pass

    def info(self, msg, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        pass

    def warn(self, msg, *args, **kwargs):
        pass

    def error(self, msg, *args, **kwargs):
        pass

    def exception(self, msg, *args, exc_info=True, **kwargs):
        pass

    def critical(self, msg, *args, **kwargs):
        pass


def create_logger(log_dir, local_rank=-1):
    if local_rank not in [-1, 0]:  # dist and not the main worker, return dummy logger
        return DummyLogger()

    logger.remove()  # clear default sys.stderr
    logger.add(sys.stderr,
               format="{time} {level} {message}",
               level="INFO")
    if log_dir is not None:
        logger.add(os.path.join(log_dir, "log.txt"),
                   format="{time} {level} {message}",
                   level="DEBUG")

    return logger


def str2bool(v):
    """
    Boolean values for argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed: int) -> None:
    """
    seed everything
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # deterministic algo, slower
    torch.backends.cudnn.benchmark = True  # speed up when fixed computation graph


def prepare_cuda():
    print('init cuda...')
    torch.zeros(1, device='cuda')


def save_code_to_git(commit_msg):
    cmd = 'git add -A ' + \
          '&& git commit -m ' + commit_msg
    os.system(cmd)


def wandb_init(args):
    # tensorboard patching must be ahead of constructing SummaryWriter
    import wandb
    # record those useful hyper-params
    wandb.init(project='576_Proj',
               entity='godkillerxiao',
               config=vars(args),
               name='-'.join(args.log_dir.split('/')[-2:])
               )
