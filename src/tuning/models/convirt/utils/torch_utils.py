"""
Utility functions for torch.
"""

import torch
from torch import nn, optim
from torch.optim import Optimizer

from transformers import AdamW

def get_optimizer(name, parameters, lr, beta1=0.9, beta2=0.999, eps=1e-8, l2=0, momentum=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=l2, momentum=momentum)
    elif name in ['adagrad', 'myadagrad']:
        # use my own adagrad to allow for init accumulator value
        return MyAdagrad(parameters, lr=lr, init_accu_value=0.1, weight_decay=l2)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=l2, betas=(beta1, beta2), eps=eps)
    elif name == 'adamw':
        return AdamW(parameters, lr=lr, weight_decay=l2, betas=(beta1, beta2), eps=eps)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=l2)
    elif name == 'adadelta':
        return torch.optim.Adadelta(parameters, lr=lr, weight_decay=l2)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def get_lrs(optimizer):
    lrs = []
    for group in optimizer.param_groups:
        lrs += [group['lr']]
    return lrs


def flatten_indices(seq_lens, width):
    flat = []
    for i, l in enumerate(seq_lens):
        for j in range(l):
            flat.append(i * width + j)
    return flat
