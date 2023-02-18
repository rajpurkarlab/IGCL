"""
Different NN Layers.
"""
import torch
from torch import nn

class FCLayer(nn.Module):
    """
    A basic FC classifier layer with `num_class` output classes.
    """
    def __init__(self, in_features, num_class, dropout=None):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(in_features, num_class)
        self.in_features = in_features
        self.out_features = num_class
        if dropout is not None and dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        if self.dropout is not None:
            return self.fc(self.dropout(x))
        else:
            return self.fc(x)

class MultiLabelBinaryFCLayer(nn.Module):
    """
    A multi-label FC classifier layer with `num_label` output heads and each a binary output.
    """
    def __init__(self, in_features, num_label, dropout=None):
        super(MultiLabelBinaryFCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_features, num_label), nn.Sigmoid())
        self.in_features = in_features
        self.out_features = num_label
        if dropout is not None and dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        if self.dropout is not None:
            return self.fc(self.dropout(x))
        else:
            return self.fc(x)
