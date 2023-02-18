# NO LONGER USE THIS FILE
import subprocess
import numpy as np
import os
import sys
import pandas as pd
from PIL import Image
import h5py
import matplotlib.pyplot as plt
import random
from typing import Optional, Callable, List

import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode

import sklearn
from sklearn.model_selection import train_test_split


class _CXRTestDataset(Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        img_path: Path to hdf5 file containing images.
        label_path: Path to file containing labels 
        cxr_labels: List of possible labels (there should be 14 for CheXpert).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, dataset_name, img_path, label_path, cxr_labels, transform=None, cutlabels=True):
        super().__init__()
        self.name = dataset_name
        self.img_dset = h5py.File(img_path, 'r')['cxr']
        full_labels = pd.read_csv(label_path)
        print("FULL LABELS", full_labels.columns)
        if cutlabels:
            full_labels = full_labels.loc[:, cxr_labels]
        else: 
            full_labels.drop(full_labels.columns[0], axis=1, inplace=True)
        self.labels = full_labels.to_numpy()
        self.transform = transform
            
    def __len__(self):
        return len(self.img_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.img_dset[idx] # np array, (320, 320)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img) # torch, (320, 320)
        
        if self.transform:
            img = self.transform(img)
            
        label = torch.from_numpy(self.labels[idx])
        sample = {'img': img, 'label': label }
    
        return sample
    
    # print("To save your dataset into a .pt file, use the save method. \n Then load the dataset with the load_CXR method. \n")

    def save(self, PATH):
        print("Saving dataset to ", PATH, " ...")
        torch.save(self, PATH+self.name+".pt")
        print("Saved pt file...")


class _TorchDataset(Dataset):
    """For converting from h5 to PyTorch datasets"""
    def __init__(self, dset: CXRTestDataset, indices = None):
        if indices is None:
            indices = range(len(dset))
        
        self.dset = dset
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image = self.dset[self.indices[idx]]['img']
        label = self.dset[self.indices[idx]]['label']
        
        label = torch.nan_to_num(label, nan=0.)  # replace blank with 0
        label = torch.abs(label)  # replace -1 (uncertain) with 1 (present)
        sample = (image, label)
        return sample
    

def _load_chexpert(cxr_labels, batch_size, pretrained=True, train_percent: Optional[float] = 0.01, K: Optional[int] = None):
    if pretrained:
        input_resolution = 224  # for pretrained model
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
        ])
    else:
        input_resolution = 320
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        ])

    chexpert_test = _CXRTestDataset('/deep/group/data/med-data/test_cxr.h5',
                                   '/deep/group/data/med-data/final_paths.csv',
                                    cxr_labels,
                                    transform,
                                  )
    chexpert_train = _CXRTestDataset('/deep/group/img-graph/CheXpert/train.h5',
                                    '/deep/group/data/med-data/train.csv',
                                    cxr_labels,
                                    transform,
                                   )
    chexpert_val = _CXRTestDataset('/deep/group/img-graph/CheXpert/valid.h5',
                                  '/deep/group/data/CheXpert-320x320/valid.csv',
                                  cxr_labels,
                                  transform,
                                 )
    
    indices = range(len(chexpert_train))
    
    # few-shot learning
    if K is not None:
        condition_count = 0  # count the number of images for a given condition found so far
        train_indices = []
        N = len(chexpert_train)

        for j in range(len(cxr_labels)):
            condition_count = 0
            while condition_count < K:
                i = random.randint(0, N-1)
                if i in train_indices:  # don't reuse indices
                    continue
                
                # This means condition is present
                if chexpert_train[i]['label'][j].item() == 1:
                    train_indices.append(i)
                    condition_count += 1
    else:
        train_indices, _ = train_test_split(indices, test_size=1-train_percent, random_state=42)
    
    train_dset = _TorchDataset(chexpert_train, train_indices)
    val_dset = _TorchDataset(chexpert_val)
    test_dset = _TorchDataset(chexpert_test)
    
    train_dataloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader
