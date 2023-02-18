import sys
sys.path.append('./')

from dracon_mod import DRACON
from model import CLIP, FinetunedModel, LinearModel, VisualTransformer, HuggingFaceImageEncoder
import torch
import warnings

def _dracon(device=torch.device("cpu"),freeze_backbone=True,num_out=1):
    # --------------------------------------------
    #  DRACON adding last layer to VISUAL model
    # --------------------------------------------

    if device==torch.device("cpu"):
        warnings.warn("Loading model on CPU.... Use GPU if available for faster training! pass device variable in chexzero function as torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') to use gpu if available")
    # assert linear_layer_dim>0, "Linear Layer Dimension has to be greater than 0!"
    assert num_out>0, "Number of classes output has to be greater than 0!"
    assert isinstance(freeze_backbone,bool), "freeze_backbone can only be a bool (True/False) value"
    assert isinstance(device,torch.device), "device has to be a type torch.device, use torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') before calling chexzero function"

    PATH = "./pretrained/igcl_dracon.pt"
    embed_dim=768
    pretrained=True
    node_features=772
    edge_features=4
    graph_layers=2
    graph_hidden=512

    trans_layers=1
    fc_layers=1
    attn_heads=2
    use_pool=True
        
    gnn = DRACON(node_features, graph_hidden,
                 edge_features, graph_layers,
                 trans_layers, fc_layers,
                 attn_heads, use_pool)
    
    visual = HuggingFaceImageEncoder(device)
    # hardcode image_encoder_dim for pretrained model
    model = CLIP(embed_dim=embed_dim, image_encoder=visual,
                 image_encoder_dim=768, graph_encoder=gnn, graph_encoder_dim=graph_hidden)
    
    model.load_state_dict(torch.load(PATH, map_location=device))


    if freeze_backbone:
        downstream_model = LinearModel(model, num_out)
        downstream_model.name = "dracon-linear"
        
    else:
        downstream_model = FinetunedModel(model, num_out)
        downstream_model.name = "dracon-finetune"

    return downstream_model.to(device)



