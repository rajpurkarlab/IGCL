import torch
from torchvision import models, transforms
import warnings


def _convirt(device=torch.device("cpu"),freeze_backbone=True,num_out=1):

    if device==torch.device("cpu"):
        warnings.warn("Loading model on CPU.... Use GPU if available for faster training! pass device variable in convirt function as torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') to use gpu if available")
    assert num_out>0, "Number of classes output has to be greater than 0!"
    assert isinstance(freeze_backbone,bool), "freeze_backbone can only be a bool (True/False) value"
    assert isinstance(device,torch.device), "device has to be a type torch.device, use torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') before calling convirt function"

    PATH = './pretrained/convirt_chest_mimic.pt'

    model = models.resnet50(pretrained=True).to(device)
   
    # print(k0)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential()
    
    model.fc = torch.nn.Linear(num_ftrs, num_out)
    model.fc = model.fc.to(device)
    model.load_state_dict(torch.load(PATH, map_location=device),strict=False)


    if freeze_backbone:
        for param in list(model.parameters())[:-2]:
            param.requires_grad = False
    
    if freeze_backbone:
        model.name = "convirt-linear"
    else:
        model.name = "convirt-finetune"

    return model.to(device)
