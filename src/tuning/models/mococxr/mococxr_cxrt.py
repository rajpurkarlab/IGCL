from .finetune.saver import ModelSaver
import torch
import torchvision
import warnings 
from .finetune.models import models
from .finetune.models import PretrainedModel 


__MODELS = {"resnet18":"tuning/pretrained/moco-cxr_resnet18.pth.tar","resnet50":"tuning/pretrained/moco-cxr_resnet50.pth.tar"}


class _model_args():
    def __init__(self,moco,model,freeze_backbone,ckpt_path):
        super(_model_args, self).__init__()
        self.moco = True
        self.model = model
        self.pretrained = False
        self.fine_tuning = freeze_backbone
        self.model_uncertainty = False
        self.ckpt_path = ckpt_path

def _load_model(ckpt_path, device, model_args, num_out, freeze_backbone,model_name):
        """Load model parameters from disk.

        Args:
            ckpt_path: Path to checkpoint to load.
            gpu_ids: GPU IDs for DataParallel.
            model_args: Model arguments to instantiate the model object.

        Returns:
            Model loaded from checkpoint, dict of additional
            checkpoint info (e.g. epoch, metric).
        """

        if model_name=="resnet18":
            model = torchvision.models.resnet18(pretrained=True)
            cl = 512
        elif model_name=="resnet50":
            model = torchvision.models.resnet50(pretrained=True)
            cl = 2048

        ckpt_dict = torch.load(ckpt_path, map_location=device)

        if not model_args.moco:
            model_fn = models.__dict__[ckpt_dict['model_name']]
        else:
            s = ckpt_dict['arch']
            model_fn = models.__dict__[s]

        tasks = list(range(20))
        model = model_fn(tasks, model_args)

        if not model_args.moco:
            model.load_state_dict(ckpt_dict['model_state'])
        else:
            state_dict = ckpt_dict['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    # state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                elif 'encoder_k' in k or 'module.queue' in k:
                    del state_dict[k]
                elif k.startswith('module.encoder_q.fc'):
                    # if 'fc.0' not in k:
                    #     state_dict['module.model.fc' + k[len("module.encoder_q.fc.2"):]] = state_dict[k]
                    # TODO: JBY these are bad
                    del state_dict[k]

            model.load_state_dict(state_dict, strict=False)

        model.fc = torch.nn.Linear(cl, num_out)
        model.fc = model.fc.to(device)

        if freeze_backbone:
            for param in list(model.parameters())[:-2]:
                param.requires_grad = False

        model = model.to(device)
        
        return model

def _mococxr(model="resnet50",device=torch.device("cpu"),freeze_backbone=True,num_out=1):

    if device==torch.device("cpu"):
        warnings.warn("Loading model on CPU.... Use GPU if available for faster training! pass device variable in mococxr function as torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') to use gpu if available")
    assert model in ["resnet50","resnet18"],"Supported model value functions are resnet18 and resnet50 only."
    assert num_out>0, "Number of classes output has to be greater than 0!"
    assert isinstance(freeze_backbone,bool), "freeze_backbone can only be a bool (True/False) value"
    assert isinstance(device,torch.device), "device has to be a type torch.device, use torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') before calling mococxr function"

    key = model
    ckpt = __MODELS[key]
    model_params = _model_args(moco=True,model=model,freeze_backbone=freeze_backbone,ckpt_path=ckpt)
    model = _load_model(ckpt_path=ckpt,device = device,model_args=model_params,num_out=num_out,freeze_backbone=freeze_backbone,model_name=model)
    model = model.to(device)
    if freeze_backbone:
        model.name = "mococxr-linear"
    else:
        model.name = "mococxr-finetune"

    return model






