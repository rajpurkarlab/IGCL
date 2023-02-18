import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from tqdm.notebook import tqdm
from typing import Optional, Callable, List

from PIL import Image
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from radgraph_dataset import RadGraphDataset
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, InterpolationMode

def rescale(img, desired_size=320):
    old_size = img.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    img = img.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it
    new_img = Image.new('L', (desired_size, desired_size))
    new_img.paste(img, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    new_img = ToTensor()(new_img)  # (1, 320, 320)
    new_img = np.repeat(new_img, 3, axis=0)  # (1, 320, 320) --> (3, 320, 320)
    new_img = torch.unsqueeze(new_img, 0)
    return new_img


class TextImageGraphDataset(Dataset):
    def __init__(
        self,
        name: str,
        text_root: str,
        image_root: str,
        graph_root: str,
        image_transform: Optional[Callable] = None,
        image_pre_transform: Optional[Callable] = None,
        graph_transform: Optional[Callable] = None,
        graph_pre_transform: Optional[Callable] = None,
        graph_pre_filter: Optional[Callable] = None,
        use_node_attr: bool = False,
        use_edge_attr: bool = False,
        cleaned: bool = False,
    ):
        self.name = name
        self.text_root = text_root
        self.image_root = image_root
        self.graph_root = graph_root
        self.image_transform = image_transform
        self.image_pre_transform = image_pre_transform
        self.graph_dset = RadGraphDataset(graph_root, name, graph_transform, graph_transform, graph_pre_filter,
                                          use_node_attr, use_edge_attr, cleaned)
        super().__init__(graph_root, graph_transform, graph_pre_transform, graph_pre_filter)
        
    
    @property
    def raw_dir(self) -> str:
        return self.graph_dset.raw_dir
        return os.path.join(self.graph_root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return self.graph_dset.processed_dir
        return os.path.join(self.graph_root, self.name, 'processed')
    
    @property
    def raw_file_names(self) -> List[str]:
        return self.graph_dset.raw_file_names
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]
    
    @property
    def processed_file_names(self) -> List[str]:
        return self.graph_dset.processed_file_names
        return 'data.pt'
    
    def download(self):
        pass
    
    def process(self):
        # Processing done in RadGraphDataset
        pass
    
    def len(self):
        return self.graph_dset.len()
    
    def get(self, idx: int) -> Data:
        """
        Args:
            idx: index of image-graph pair to retrieve
        """
        # Load graph
        graph = self.graph_dset[idx]
        
        # Lazily load corresponding image to prevent loading too many images into memory at once
        image = Image.new("RGB", (320, 320))  # default blank image
        image_dir = f'{self.image_root}/{graph.pid[:-4]}'
        if not os.path.isdir(image_dir):
            print(f'ERROR: image directory {image_dir} does not exist! Skipping...')
        else:
            # Just take the first image we find
            for filename in os.listdir(image_dir):
                if not filename.endswith('.jpg'): continue
                image_path = os.path.join(image_dir, filename)
                image = Image.open(image_path)
                break
            
        if self.image_pre_transform:
            image = self.image_pre_transform(image)
        if type(image) != type(torch.tensor(0)):
            image = ToTensor()(image)
        if self.image_transform:
            image = self.image_transform(image)
        
        # Set image as a graph-level attribute
        graph.image = image
        
        # Lazily load corresponding entity tokens
        tokens = []
        tokens_file = os.path.join(self.text_root, graph.pid)
        if not os.path.isfile(tokens_file):
            print(f'ERROR: tokens file {tokens_file} does not exist! Skipping...')
        else:
            with open(tokens_file, 'r') as f:
                tokens = f.readlines()
        
        if len(tokens) != graph.x.shape[0]:
            print(f'ERROR: len(tokens) = {len(tokens)} != {graph.x.shape[0]} = graph.x.shape[0]')
            
        # # Set tokens as graph-level attribute
        graph.tokens = tokens
        
        # # Get BERT embeddings
        # inputs = self.tokenizer(tokens, padding=True, return_tensors="pt")
        # outputs = self.encoder(**inputs)
        # pooled_output = outputs.pooler_output
        # graph.token_embeddings = pooled_output
            
        return graph