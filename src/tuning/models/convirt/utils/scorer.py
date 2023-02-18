"""
Scoring utilities.
"""

from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def get_accuracy(y_true, y_pred):
    """
    Calculate accuracy based on labels and predictions in iterables.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 2: # input is probabilities, convert to labels
        y_pred = np.argmax(y_pred, axis=1)
    assert len(y_true) == len(y_pred), "Prediction must have equal length as key."
    accu = accuracy_score(y_true, y_pred)
    return accu

def get_auc_score(y_true, y_pred):
    """
    Calculate ROC-AUC based on labels and predictions.

    Args:
        y_true: a list of true labels
        y_pred: a list of pred probabilities
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_true)) == 2:
        # convert to prob of the positive label
        if y_pred.ndim == 2 and y_pred.shape[1] == 2:
            y_pred = y_pred[:,1]
        elif y_pred.ndim == 1:
            pass
        else:
            raise Exception(f"Unsupported prediction vector size: {y_pred.shape}")
        auc = roc_auc_score(y_true, y_pred)
    else:
        auc = roc_auc_score(to_categorical(y_true), y_pred, average='macro')
    return auc

def get_f1_score(y_true, y_pred, average='macro'):
    """
    Calculate macro-F1 score based on labels and predictions.

    Args:
        y_true: a list of true labels
        y_pred: a list of pred probabilities
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_true)) == 2:
        if y_pred.ndim == 1:
            # thresholding to binary output labels
            y_pred = (y_pred > 0.5).astype('int')
        else: # convert to labels with argmax
            y_pred = np.argmax(y_pred, axis=1)
        f1 = f1_score(y_true, y_pred)
    else:
        f1 = f1_score(y_true, np.argmax(y_pred, axis=1), average=average)
    return f1

def coalesce_predictions(y_pred, y_group):
    """
    Coalesce example-level predictions into group-level.
    """
    assert len(y_pred) == len(y_group), "Predictions and their group counts do not match."
    group2preds = defaultdict(list)
    for p, g in zip(y_pred, y_group):
        group2preds[g].append(p)
    # coalesce
    group2prob = dict()
    for g, preds in group2preds.items(): 
        mean_prob = np.mean(np.asarray(preds), axis=0)
        group2prob[g] = mean_prob
    return group2prob

def to_categorical(y, num_classes=None):
    """
    Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    Arguments:
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    Returns:
        A binary matrix representation of the input. The classes axis is placed
        last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
      input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
      num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical