"""
Functions and modules for criterions and losses.
"""
import numpy as np
import torch
from torch import nn

def get_weighted_cross_entropy_loss(all_labels=None, log_dampened=False):
    if all_labels is None:
        return nn.CrossEntropyLoss()
    
    if isinstance(all_labels, list):
        all_labels = np.array(all_labels)
    _, weights = np.unique(all_labels, return_counts=True)
    weights = weights / float(np.sum(weights))
    weights = 1. / weights
    if log_dampened:
        weights = 1 + np.log(weights)
    weights /= np.sum(weights) # normalize
    crit = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights).type('torch.FloatTensor')
    )
    return crit
