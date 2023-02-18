"""
Load and modify image models.
"""

import logging
import torchvision
import torch
from torch import nn

logger = logging.getLogger('transfer')

def densenet(pretrained=True):
    """ Make densenet121 a default densenet. """
    return densenet121(pretrained)

def densenet121(pretrained=True):
    model = torchvision.models.densenet121(pretrained=pretrained)
    dim_feats = model.classifier.in_features
    # model.features.norm5 = nn.Sequential(nn.AdaptiveAvgPool2d(7), nn.BatchNorm2d(dim_feats, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    return model, dim_feats

def densenet161(pretrained=True):
    model = torchvision.models.densenet161(pretrained=pretrained)
    dim_feats = model.classifier.in_features
    # model.features.norm5 = nn.Sequential(nn.AdaptiveAvgPool2d(7), nn.BatchNorm2d(dim_feats, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    return model, dim_feats

def resnet(pretrained=True):
    """ Make resnet101 a default resnet. """
    return resnet101(pretrained)

def resnet50(pretrained=True):
    model = torchvision.models.resnet50(pretrained=pretrained)
    dim_feats = model.fc.in_features
    return model, dim_feats

def resnet101(pretrained=True):
    model = torchvision.models.resnet101(pretrained=pretrained)
    dim_feats = model.fc.in_features
    return model, dim_feats

def resnet152(pretrained=True):
    model = torchvision.models.resnet152(pretrained=pretrained)
    dim_feats = model.fc.in_features
    return model, dim_feats

def get_classifier(model):
    if hasattr(model, 'classifier'):
        return model.classifier
    elif hasattr(model, 'fc'):
        return model.fc
    else:
        raise Exception(f"Cannot find classifier layer for model: neither 'fc' nor 'classifier' layer exists.")

def set_classifier(model, new_clf_layer):
    """
    Set the last classifier layer in the model to a new classifier layer.
    """
    if hasattr(model, 'classifier'):
        # assert model.classifier.in_features == new_clf_layer.in_features, \
            # "in_features does not match between old and new classifier layer."
        model.classifier = new_clf_layer
    elif hasattr(model, 'fc'):
        # assert model.fc.in_features == new_clf_layer.in_features, \
            # "in_features does not match between old and new classifier layer."
        model.fc = new_clf_layer
    else:
        raise Exception(f"Cannot set classifier layer for model: neither 'fc' nor 'classifier' layer exists.")

def drop_classifier(model):
    """
    Drop the last classifier layer in the model and output the CNN features directly.
    """
    set_classifier(model, nn.Identity())

def reload_from_pretrained(model, filename, key=None, strict=True):
    """
    Load model weights from pretrained model weights in file.

    Args:
        model: the model to load weights to
        filename: the checkpoint file where weights are stored
        key: a key to use (e.g., 'image_encoder') if the state dict is stored as a keyed item in the file
        strict: whether to enforce strict name mapping
    """
    state_dict = torch.load(filename)
    if key is not None and key.lower() != "none" and len(key) > 0:
        state_dict = state_dict[key]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    if len(missing_keys) > 0:
        logger.info(f"Missing keys from pretrained file: {missing_keys}")
    if len(unexpected_keys) > 0:
        logger.info(f"Unexpected keys from pretrained file: {unexpected_keys}")
    return model