"""
Handling labels for different tasks.
"""
import numpy as np

def convert_3class_label_to_binary(pred):
    """
    Convert the 3class priority label (0-2) to binary label space.

    Args:
        - pred: a list of integer labels
    """
    res = []
    for p in pred:
        if p >= 1:
            res += [1]
        else:
            res += [0]
    return res

def convert_3class_probability_to_binary(pred):
    """
    Convert the 3class priority probabilities to probabilities in binary space.

    Args:
        - pred: a numpy array of dimension N x 3
    """
    N, d = pred.shape
    assert d == 3, "Input must have 3 probability scores at each row."
    res = np.zeros([N, 2])
    res[:, 0] = pred[:, 0]
    res[:, 1] = pred[:, 1] + pred[:, 2]
    return res

