# -*- coding:UTF-8 -*-
import torch
import horovod.torch as hvd
import logging


def init_distributed():
    hvd.init()


def pin_gpu_to_process():
    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())


def wrap_hvd_optimizer(optimizer, named_params, use_adasum):
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=named_params,
                                         op=hvd.Adasum if use_adasum else hvd.Average)
    return optimizer


def broadcast_model_params(model, optimizer):
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)


def get_world_size():
    return hvd.size()


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def setup_gradient_scaler():
    scaler = torch.cuda.amp.GradScaler()
    return scaler


class LogMessage:
    def __init__(self, is_distributed):
        self.is_distributed = is_distributed

    def __call__(self, message):
        if (self.is_distributed and hvd.rank() == 0) or not self.is_distributed:
            logging.info(message)
