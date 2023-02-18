import os

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from typing import Optional, Callable, List

from PIL import Image
import h5py

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.data.separate import separate
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, InterpolationMode

from radgraph_dataset import RadGraphDataset

class ImageGraphDataset(InMemoryDataset):
    def __init__(
        self,
        name: str,
        graph_root: str,
        raw_image_path: str,
        processed_graph_path: str,
        processed_image_path: str,
        image_transform: Optional[Callable] = None,
        graph_transform: Optional[Callable] = None,
        graph_pre_transform: Optional[Callable] = None,
        graph_pre_filter: Optional[Callable] = None,
        use_node_attr: bool = False,
        use_edge_attr: bool = False,
        cleaned: bool = False,
    ):
        self.name = name
        self.graph_root = graph_root
        self.raw_image_path = raw_image_path
        
        self.processed_graph_path = processed_graph_path
        self.processed_image_path = processed_image_path
        
        self.image_transform = image_transform
        self.graph_transform = graph_transform
        
        self.raw_image_dset = h5py.File(raw_image_path, 'r')['cxr_unprocessed']
        # No graph_transform since we handle that here
        self.raw_graph_dset = RadGraphDataset(graph_root, name, None, graph_pre_transform, graph_pre_filter,
                                          use_node_attr, use_edge_attr, cleaned)
        
        super().__init__(graph_root, None, graph_pre_transform, graph_pre_filter)
        
        self.img_dset = h5py.File(processed_image_path, 'r')['cxr']
        self.graph_data, self.graph_slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_dir(self) -> str:
        return self.raw_graph_dset.processed_dir

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.graph_root, 'MIMIC')
    
    @property
    def raw_file_names(self) -> List[str]:
        return self.raw_graph_dset.processed_file_names
    
    @property
    def processed_file_names(self) -> List[str]:
        return [self.processed_graph_path, self.processed_image_path]
    
    @property
    def num_features(self) -> int:
        return self.raw_graph_dset.num_features
    
    def download(self):
        pass
    
    def process(self):
        print('STARTING PROCESSING')
        TEXT_ROOT = '/deep/group/data/med-data/train_unprocessed.csv'
        df = pd.read_csv(TEXT_ROOT)
        df['path_processed'] = df.Path.apply(lambda x: '/'.join(x.split('/')[8:11]) + '.txt')
        
        # Find the indices of images corresponding to valid graphs
        pid_map = {}
        for i in tqdm(range(len(self.raw_graph_dset))):
            pid_map[self.raw_graph_dset[i].pid] = i
        
        good_image_indices = []
        good_graph_pids = []
        good_graph_indices = []
        graph_list = []
        
        for i in tqdm(range(len(self.raw_image_dset))):
            pid = df.iloc[i].path_processed
            if pid not in pid_map:  # not a good image
                continue
            
            good_image_indices.append(i)
            good_graph_pids.append(pid)
            good_graph_indices.append(pid_map[pid])
            
            graph_list.append(self.raw_graph_dset[pid_map[pid]])
            # print("GRAPH", self.raw_graph_dset[pid_map[pid]])
        
        print("NUMBER OF GOOD IMAGES", len(good_image_indices))
        # Save indices of aligned image-graph pairs to a csv file
        aligned = pd.DataFrame({'img_idx': good_image_indices, 'graph_pid': good_graph_pids, 'graph_idx': good_graph_indices})
        print('SAVING IMG-GRAPH INDICES TO CSV')
        aligned.to_csv('/deep/group/img-graph/MIMIC/img_graph_metadata.csv', index=False)
        
        # Make graph dataset
        if self.pre_filter is not None:
            graph_list = [data for data in graph_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(data) for data in graph_list]
        
        data, slices = self.collate(graph_list)
        torch.save((data, slices), self.processed_paths[0])
        
        # Make image dataset
        with h5py.File(self.processed_image_path,'w') as h5f:
            img_dset = h5f.create_dataset('cxr', shape=(len(good_image_indices), 320, 320))
            print('Image dataset initialized.')

            for i in tqdm(range(len(good_image_indices))):
                img = self.raw_image_dset[good_image_indices[i]]
                img_dset[i] = img
        
        print('DONE PROCESSING')
        
    def len(self):
        return len(self.img_dset)
    
    def get(self, idx: int) -> Data:
        """
        Args:
            idx: index of image-graph pair to retrieve
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if self.len() == 1:
            graph = copy.copy(self.data)

        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            graph = copy.copy(self._data_list[idx])
        
        # Get graph
        graph = separate(
            cls=self.graph_data.__class__,
            batch=self.graph_data,
            idx=idx,
            slice_dict=self.graph_slices,
            decrement=False,
        )
        # print("PRE TRANSFORM GRAPH", graph.num_nodes, graph)
        
        if self.graph_transform:
            graph = self.graph_transform(graph)
            # print("POST TRANSFORM GRAPH", graph.num_nodes, graph)
        self._data_list[idx] = copy.copy(graph)
        
        # Get image
        img = self.img_dset[idx] # np array, (320, 320)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img) # torch, (3, 320, 320)
        if self.image_transform:
            img = self.image_transform(img)
        
        sample = {'img': img, 'graph': graph}
        return sample