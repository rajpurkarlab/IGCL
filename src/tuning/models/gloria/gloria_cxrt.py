from . import models
import torch
import copy
import warnings

from .models.gloria_model import GLoRIA
from .models.vision_model import PretrainedImageClassifier


_MODELS = {
    "resnet50": "tuning/pretrained/gloria_chexpert_resnet50.ckpt",
    "resnet18": "tuning/pretrained/gloria_chexpert_resnet18.ckpt",
}

def _build_gloria_model(cfg):
    gloria_model = GLoRIA(cfg)
    return gloria_model



def _build_gloria_from_ckpt(ckpt,device):

    ckpt = torch.load(ckpt,map_location=device)
    cfg = ckpt["hyper_parameters"]
    ckpt_dict = ckpt["state_dict"]

    fixed_ckpt_dict = {}
    for k, v in ckpt_dict.items():
        new_key = k.split("gloria.")[-1]
        fixed_ckpt_dict[new_key] = v
    ckpt_dict = fixed_ckpt_dict

    gloria_model = _build_gloria_model(cfg)
    gloria_model.load_state_dict(ckpt_dict)

    return gloria_model

def _gloria(model="resnet50",device=torch.device("cpu"),freeze_backbone=True,num_ftrs=2048,num_out=1):
    if device==torch.device("cpu"):
        warnings.warn("Loading model on CPU.... Use GPU if available for faster training! pass device variable in gloria function as torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') to use gpu if available")
    assert model in ["resnet50","resnet18"],"Supported model value functions are resnet18 and resnet50 only."
    assert num_ftrs>0, "num_ftrs (Number of features) variable has to be greater than 0!"
    assert num_out>0, "Number of classes output has to be greater than 0!"
    assert isinstance(freeze_backbone,bool), "freeze_backbone can only be a bool (True/False) value"
    assert isinstance(device,torch.device), "device has to be a type torch.device, use torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') before calling gloria function"

    gloria_model = _build_gloria_from_ckpt(_MODELS[model],device)
    img_model = copy.deepcopy(gloria_model.img_encoder)
    del gloria_model

    model = PretrainedImageClassifier(img_model,num_out,num_ftrs,freeze_backbone)
    if freeze_backbone:
        model.name = "gloria-linear"
    else:
        model.name = "gloria-finetune"
    return model.to(device)
