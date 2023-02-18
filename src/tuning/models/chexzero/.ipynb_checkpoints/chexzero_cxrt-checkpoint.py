import clip
import torch
import warnings

# ---------------------------------------------------------------
#  Code adapted from the CheXZero paper - Loads the CheXZero model 
#  with the CLIP architecture.
#  Good to know: need to install PyTorch 1.7.1 or higher 
#  see https://github.com/openai/CLIP/blob/main/README.md
# ---------------------------------------------------------------
    
def _load_clip(model_path,device=None):
    '''
    FUNCTION: load_clip
    -------------------------------
    This function loads in a model with the CLIP model 
    architecture. 
    
    args: 
        * model_path - path to model weights that the model
        will be initialized with 
    '''    
    # set device 
    # load clip pre-trained model
    print("Loading the CLIP architecture...")
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    
    # if a model_path is provided, load in weights to backbone
    if model_path != None: 
        print("Loading the pretrained model from path...")
        ckpt_in = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt_in,strict=True)

    return model, preprocess


def _chexzero(device=torch.device("cpu"),freeze_backbone=True,linear_layer_dim=512,num_out=1):
    # --------------------------------------------
    #  cheXZero adding last layer to VISUAL model
    # --------------------------------------------

    if device==torch.device("cpu"):
        warnings.warn("Loading model on CPU.... Use GPU if available for faster training! pass device variable in chexzero function as torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') to use gpu if available")
    assert linear_layer_dim>0, "Linear Layer Dimension has to be greater than 0!"
    assert num_out>0, "Number of classes output has to be greater than 0!"
    assert isinstance(freeze_backbone,bool), "freeze_backbone can only be a bool (True/False) value"
    assert isinstance(device,torch.device), "device has to be a type torch.device, use torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') before calling chexzero function"

    PATH = "./pretrained/chexzero.pt"
    cheXZero, preprocess = _load_clip(PATH)
    linear = torch.nn.Linear(linear_layer_dim, num_out)
    model = torch.nn.Sequential(cheXZero.visual.float(), linear).to(device)


    if freeze_backbone:
        for param in list(model.parameters())[:-2]:
            param.requires_grad = False
    if freeze_backbone:
        model.name = "chexzero-linear"
    else:
        model.name = "chexzero-finetune"

    return model.to(device)



