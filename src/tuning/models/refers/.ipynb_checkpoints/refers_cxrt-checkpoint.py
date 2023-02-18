from . import models
import torch
import copy
import warnings
from .models.modeling import VisionTransformer, CONFIGS



def _load_weights(model, weight_path,device):
    pretrained_weights = torch.load(weight_path, map_location=device)
    model_weights = model.state_dict()

    load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}

    model_weights.update(load_weights)
    model.load_state_dict(model_weights)
    return model


def _setup_REFERS(img_size,num_classes,pretrained_dir,device):
    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(config,img_size,zero_head=True,num_classes=num_classes)
    model = _load_weights(model,pretrained_dir,device)
    model.to(device)

    return model

class _PretrainedREFER(torch.nn.Module):
    def __init__(
        self,
        model_in: torch.nn.Module,
        freeze_encoder: bool = True,
    ):
        super(_PretrainedREFER, self).__init__()
        self.model_in = model_in
        if freeze_encoder:
            for param in list(self.model_in.parameters())[:-2]:
                param.requires_grad = False

    def forward(self, x):
        pred = self.model_in(x)
        return pred


def _refers(device=torch.device("cpu"),freeze_backbone=True,num_out=1):
    if device==torch.device("cpu"):
        warnings.warn("Loading model on CPU.... Use GPU if available for faster training! pass device variable in refers function as torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') to use gpu if available")
    assert num_out>0, "num_out (Number of classes output) has to be greater than 0!"
    assert isinstance(freeze_backbone,bool), "freeze_backbone can only be a bool (True/False) value"
    assert isinstance(device,torch.device), "device has to be a type torch.device, use torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') before calling refers function"


    refers_model = _setup_REFERS(img_size=224,num_classes=num_out,pretrained_dir="./pretrained/refers_checkpoint.pth",device=device)
    model = _PretrainedREFER(refers_model,freeze_backbone)
    del refers_model
    if freeze_backbone:
        model.name = "refers-linear"
    else:
        model.name = "refers-finetune"

    return model.to(device)
