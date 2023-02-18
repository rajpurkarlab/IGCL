import subprocess
import numpy as np
import os
import sys
import pandas as pd
from PIL import Image
import h5py
import matplotlib.pyplot as plt
from typing import Optional, Callable, List

import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode

import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from model import VisualTransformer, HuggingFaceImageEncoder, CLIP
from model import MyGCN, MyGAT, MyRGCN, GlobalAttentionNet, NoEdgeAttrGAT
from dracon import DRACON

def load_clip(num_node_features=4, num_edge_features=3, model_path=None, pretrained=False, config=None):
    '''
    FUNCTION: load_clip
    -------------------------------
    This function loads in a model with the CLIP model 
    architecture. 
    
    args: 
        * model_path (optional) - path to model weights that the model
        will be initialized with 
        * pretrained (optional) - if True, will load the pretrained 
        CLIP model
        * config (optional) - overrides default hyperparams
    '''

    params = {
        'embed_dim':768,
        'image_resolution': 320,
        'vision_layers': 12,
        'vision_width': 768,
        'vision_patch_size': 16,
        
        'node_features': num_node_features,
        'edge_features': num_edge_features,
        'graph_layers': 3,
        'graph_hidden': 128,
    }
    
    # Override default hyperparams if config is specified
    if config is not None:
        params.update(config)
        
    print('NUM GRAPH LAYERS!!!!', params['graph_layers'])
    
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if params['graph_architecture'] == 'MyGCN':
        gnn = MyGCN(params['node_features'], params['graph_layers'], params['graph_hidden'])
    elif params['graph_architecture'] == 'MyGAT':
        gnn = MyGAT(params['node_features'], params['graph_layers'],
                    params['graph_hidden'], params['edge_features'])
    elif params['graph_architecture'] == 'GlobalAttentionNet':
        gnn = GlobalAttentionNet(params['node_features'], params['graph_layers'], params['graph_hidden'])
    elif params['graph_architecture'] == 'MyRGCN':
        gnn = MyRGCN(params['node_features'], params['graph_layers'],
                     params['graph_hidden'], params['edge_features'])
    elif params['graph_architecture'] == 'NoEdgeAttrGAT':
        gnn = NoEdgeAttrGAT(params['node_features'], params['graph_layers'], params['graph_hidden'])
    elif params['graph_architecture'] == 'DRACON':
        gnn = DRACON(params['node_features'], params['graph_hidden'],
                     params['edge_features'], params['graph_layers'],
                     params['trans_layers'], params['fc_layers'],
                     params['attn_heads'], params['use_pool'])
    
    if pretrained:  # pretrained image encoder
        visual = HuggingFaceImageEncoder(device)
        # hardcode image_encoder_dim for pretrained model
        model = CLIP(embed_dim=params['embed_dim'], image_encoder=visual,
                     image_encoder_dim=768, graph_encoder=gnn, graph_encoder_dim=params['graph_hidden'])  
        print("Loaded in pretrained model.")
    else:
        vision_heads = params['vision_width'] // 64
        visual = VisualTransformer(
            input_resolution=params['image_resolution'],
            patch_size=params['vision_patch_size'],
            width=params['vision_width'],
            layers=params['vision_layers'],
            heads=vision_heads,
        )
        model = CLIP(embed_dim=params['embed_dim'], image_encoder=visual,
                     image_encoder_dim=params['vision_width'], graph_encoder=gnn, graph_encoder_dim=params['graph_hidden'])
        print("Loaded in clip model.")
    
    # if a model_path is provided, load in weights to backbone
    if model_path != None: 
        model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def save(model, path): 
    torch.save(model.state_dict(), path)