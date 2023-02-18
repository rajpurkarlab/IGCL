import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import Image
import h5py

import torch
# from torch.utils import data
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DataListLoader
from torch import nn
import torch.optim as optim
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode

import sys
sys.path.append('dataset')

from add_dummy_node import AddDummyNode
from add_meta_node import AddMetaNodes
from add_reaction_node import AddReactionNodes
from image_graph_dataset import ImageGraphDataset
from model import CLIP
from train_helper import load_clip, save


def load_data(split, graph_root, raw_image_path, processed_graph_path, processed_image_path, batch_size=4,
              pretrained=False,
              graph_transform='meta',
              dataloader='DataLoader',
              verbose=False): 
    if torch.cuda.is_available():
        dev = "cuda:0" 
        cuda_available = True
        print('Using CUDA.')
    else:  
        dev = "cpu"  
        cuda_available = False
        print('Using cpu.')
    
    device = torch.device(dev)
    
    if cuda_available: 
        torch.cuda.set_device(device)

    if pretrained:
        input_resolution = 224
        image_transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
        ])
        print('Interpolation Mode: ', InterpolationMode.BICUBIC)
        print("Finished image transforms for pretrained model.")
    else: 
        input_resolution = 320
        image_transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        ])
        print("Finished image transforms for clip model.")
    
    print("GRAPH TRANSFORM", graph_transform)
    if graph_transform == 'dummy':
        graph_transform = lambda x: AddDummyNode()(x)
    elif graph_transform == 'meta':
        graph_transform = lambda x: AddMetaNodes()(x)
    elif graph_transform == 'reaction':
        graph_transform = lambda x: AddReactionNodes()(x)
    else:
        graph_transform = None
    
    
    # torch_dset = CXRDataset(path=cxr_filepath, transform=transform)
    torch_dset = ImageGraphDataset(name=split,
                                   graph_root=graph_root,
                                   raw_image_path=raw_image_path,
                                   processed_graph_path=processed_graph_path,
                                   processed_image_path=processed_image_path,
                                   image_transform=image_transform,
                                   graph_transform=graph_transform,
                                  )
    
    if verbose: 
        for i in range(len(torch_dset)):
            sample = torch_dset[i]
            plt.imshow(sample['img'][0])
            plt.show()
            print(i, sample['img'].size(), sample['graph'])
            if i == 3:
                break
    
    num_node_features = torch_dset.num_features
    # print("NUM GRAPH FEATURES", torch_dset.num_features)
    
    loader_params = {'batch_size':batch_size, 'shuffle': True, 'num_workers': 4}
    
    # set aside 500 image-graph pairs of the training data for validation
    val_len = 500
    train_len = len(torch_dset) - val_len

    train_dset, val_dset = torch.utils.data.random_split(torch_dset, [train_len, val_len],
                                                         generator=torch.Generator().manual_seed(229))

    if dataloader == 'DataListLoader':
        train_data_loader = DataListLoader(train_dset, **loader_params)
        val_data_loader = DataListLoader(val_dset, **loader_params)
    else:
        train_data_loader = DataLoader(train_dset, **loader_params)
        val_data_loader = DataLoader(val_dset, **loader_params)
    
    return train_data_loader, val_data_loader, num_node_features, device


####################
def make(config, split, graph_root, raw_image_path, processed_graph_path, processed_image_path, model_path=None): 
    '''
    FUNCTION: make
    ---------------------------------
    This function makes the model, the data loader, loss and optimizer. 
    
    args: 
        * config - dict, configuration of experiment
        * cxr_filepath - string, filepath to chest x-ray images and BERT embeddings
        * model_path - string, filepath to previously trained model
    '''
    # train_data_loader, val_data_loader, device = load_data(cxr_filepath,
    #                                                        batch_size=config.batch_size,
    #                                                        pretrained=config.pretrained,
    #                                                       )
    
    train_data_loader, val_data_loader, num_node_features, device = load_data(split=split,
                                                                               graph_root=graph_root,
                                                                               raw_image_path=raw_image_path,
                                                                               processed_graph_path=processed_graph_path,
                                                                               processed_image_path=processed_image_path,
                                                                               batch_size=config.batch_size,
                                                                               pretrained=config.pretrained,
                                                                              graph_transform=config.graph_transform,
                                                                              dataloader=config.dataloader,
                                                                              )
    model = load_clip(num_node_features=num_node_features, model_path=model_path, pretrained=config.pretrained,
                     config=config)
    model.to(device)
    print('Model on Device.')

    # make the optimizer 
    criterion = nn.CrossEntropyLoss().cuda()
    
    if config.optimizer == "adam":
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    elif config.optimizer == "sgd": 
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
        
    return model, train_data_loader, val_data_loader, device, criterion, optimizer
    

def train_batch(images, graphs, model, device, criterion, optimizer):
    # images, graphs = images.to(device), graphs.to(device)
    images = images.to(device)
    if type(graphs) == type([]):
        graphs = [graph.to(device) for graph in graphs]
    else:
        graphs = graphs.to(device)
    
    # Forward pass ➡
    logits_per_image, logits_per_graph = model(images, graphs)
    
    # Create labels
    batch_size = images.shape[0]
    labels = torch.arange(batch_size).to(device)
    
    # Compute loss
    loss_img = criterion(logits_per_image, labels)
    loss_graph = criterion(logits_per_graph, labels)
    loss = (loss_img + loss_graph)/2 # avg. img and graph loss

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()
    
    # Step with optimizer
    optimizer.step()
        
    return loss


def train_log(loss, example_ct, epoch):
    loss = float(loss)
    # where the magic happens
    print(f"Train loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    
    
def train(model_root, model, train_loader, val_loader, device, criterion, optimizer, config, index): 
    # Run training
    total_batches = len(train_loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    report_freq = config.log_interval
    highest_mean_auc = 0 # save highest mean auc
    
    # Freeze the image encoder initially
    if config.image_freeze_interval > 0:
        print(f"FREEZING IMAGE ENCODER FOR {config.image_freeze_interval} BATCHES")
        for param in model.visual.encoder.parameters():
            param.requires_grad = False

    for epoch in range(config.epochs):
        print(f"Epoch {epoch}\n-------------------------------")
        running_loss = 0.0 # running loss over batch
        for idx, data in enumerate(tqdm(train_loader)):
            torch.cuda.empty_cache()
            
            # print("PRINTING DATA")
            # print(type(data))
            # print(len(data))
            # print(data)
            
            # get the images
            if config.dataloader == 'DataLoader':
                images = data['img']
                # print('images.shape', images.shape)

                # get the graphs (in this case BERT embeddings) 
                graphs = data['graph']
            elif config.dataloader == 'DataListLoader':
                image_list = [x['img'] for x in data]
                images = torch.stack(image_list, dim=0)
                # print('images.shape', images.shape)
                graphs = [x['graph'] for x in data]
            
            # perform step for a single batch
            loss = train_batch(images, graphs, model, device, criterion, optimizer)
            
            # Potentially unfreeze model after `image_freeze_interval` batches
            if epoch == 0 and (idx + 1) == config.image_freeze_interval:
                print(f"UNFREEZING IMAGE ENCODER AFTER {config.image_freeze_interval} BATCHES")
                for param in model.visual.encoder.parameters():
                    param.requires_grad = True

        # Epoch done; save model
        model_path = f'{config.model_root}/{config.model_name}-{epoch+1}_chkpt_{index}.pt'
        save(model, model_path)

    # Training done; save model
    model_path = f'{config.model_root}/{config.model_name}-{config.epochs}_FINAL_{index}.pt'
    save(model, model_path)

