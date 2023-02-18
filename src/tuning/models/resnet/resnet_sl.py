import torch
from torchvision import models, transforms


def _resnet(device=torch.device("cpu"),num_out=1):
    model = models.resnet50(pretrained=True).to(device)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_out)
    model.fc = model.fc.to(device)

    model.name="resnet"

    return model