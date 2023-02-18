from model import MyRGCN, CLIP, FinetunedModel, LinearModel, VisualTransformer, HuggingFaceImageEncoder
import torch
import warnings

def _rgcn(device=torch.device("cpu"),freeze_backbone=True,num_out=1):
    # --------------------------------------------
    #  RGCN adding last layer to VISUAL model
    # --------------------------------------------

    if device==torch.device("cpu"):
        warnings.warn("Loading model on CPU.... Use GPU if available for faster training! pass device variable in chexzero function as torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') to use gpu if available")
    # assert linear_layer_dim>0, "Linear Layer Dimension has to be greater than 0!"
    assert num_out>0, "Number of classes output has to be greater than 0!"
    assert isinstance(freeze_backbone,bool), "freeze_backbone can only be a bool (True/False) value"
    assert isinstance(device,torch.device), "device has to be a type torch.device, use torch.device('"'cuda:0'"' if torch.cuda.is_available() else '"'cpu'"') before calling chexzero function"

    PATH = "tuning/pretrained/igcl_rgcn.pt"
    # PATH = '/deep/u/danieljm/ImgGraph/saved_models/best_models/ViT-RGCN-dummy-attr_64_2_512_4.pt'
    embed_dim=768
    pretrained=True
    node_features=772
    edge_features=3
    graph_layers=2
    graph_hidden=512
        
    gnn = MyRGCN(node_features, graph_layers,
                 graph_hidden, edge_features)
    
    visual = HuggingFaceImageEncoder(device)
    # hardcode image_encoder_dim for pretrained model
    model = CLIP(embed_dim=embed_dim, image_encoder=visual,
                 image_encoder_dim=768, graph_encoder=gnn, graph_encoder_dim=graph_hidden)

    model.load_state_dict(torch.load(PATH, map_location=device))
    
    if freeze_backbone:
        downstream_model = LinearModel(model, num_out)
        downstream_model.name = "rgcn-linear"
        
    else:
        downstream_model = FinetunedModel(model, num_out)
        downstream_model.name = "rgcn-finetune"
        
    

    return downstream_model.to(device)



