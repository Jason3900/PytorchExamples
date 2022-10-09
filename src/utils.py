# -*- coding:UTF-8 -*-
import torch
import horovod.torch as hvd
import logging

def init_distributed():
    hvd.init()

def pin_gpu_to_process():
    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())

def wrap_hvd_optimizer(optimizer, named_params):
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=named_params)
    return optimizer

def broadcast_model_params(model, optimizer):
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

def get_world_size():
    return torch.distributed.get_world_size()

class LogMessage:
    def __init__(self, is_distributed):
        self.is_distributed = is_distributed

    def __call__(self, message):
        if (self.is_distributed and hvd.rank() == 0) or not self.is_distributed:
            logging.info(message)
