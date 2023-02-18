import os
import os.path as osp
import glob

import torch
import torch.nn.functional as F
import numpy as np
from torch_sparse import coalesce
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data

names = [
    'A', 'graph_indicator', 'node_labels', 'node_attributes',
    'edge_labels', 'edge_attributes', 'graph_labels', 'graph_attributes',
    'patient_id'
]


def read_radgraph_data(folder, prefix):
    files = glob.glob(osp.join(folder, '{}_*.txt'.format(prefix)))
    if prefix == "train":
        print("node attributes is defined out of memory for train set")
    else:
        pth_files = glob.glob(osp.join(folder, '{}_*.pth'.format(prefix)))
        files = files + pth_files
    names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]

    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1

    node_attributes = node_labels = None
    # only define in-memory node_attributes for valid and test
    if 'node_attributes' in names:
        node_attributes = read_file(folder, prefix, 'node_attributes')
    if 'node_labels' in names:
        node_labels = read_file(folder, prefix, 'node_labels', torch.long)
        if node_labels.dim() == 1:
            node_labels = node_labels.unsqueeze(-1)
        node_labels = node_labels - node_labels.min(dim=0)[0]
        node_labels = node_labels.unbind(dim=-1)
        node_labels = [F.one_hot(x, num_classes=-1) for x in node_labels]
        node_labels = torch.cat(node_labels, dim=-1).to(torch.float)
    x = cat([node_attributes, node_labels])

    edge_attributes, edge_labels = None, None
    if 'edge_attributes' in names:
        edge_attributes = read_file(folder, prefix, 'edge_attributes')
    if 'edge_labels' in names:
        edge_labels = read_file(folder, prefix, 'edge_labels', torch.long)
        if edge_labels.dim() == 1:
            edge_labels = edge_labels.unsqueeze(-1)
        edge_labels = edge_labels - edge_labels.min(dim=0)[0]
        edge_labels = edge_labels.unbind(dim=-1)
        edge_labels = [F.one_hot(e, num_classes=-1) for e in edge_labels]
        edge_labels = torch.cat(edge_labels, dim=-1).to(torch.float)
    edge_attr = cat([edge_attributes, edge_labels])

    if 'patient_id' in names:
        patient_id = read_file(folder, prefix, 'patient_id', torch.long)

    y = None
    if 'graph_attributes' in names:  # Regression problem.
        y = read_file(folder, prefix, 'graph_attributes')
    elif 'graph_labels' in names:  # Classification problem.
        y = read_file(folder, prefix, 'graph_labels', torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pid=patient_id)
    data, slices = split(data, batch)

    return data, slices


def read_node_attributes(folder, idx):
    """node attributes for a graph idx for train set

    Args:
        folder ([type]): raw folder
        idx ([type]): [description]

    Returns:
        [type]: [description]
    """
    pth_file = osp.join(folder, 'node_attributes', f'train_node_attributes-{idx}.pth')
    if osp.exists(pth_file):
        return torch.load(pth_file).cpu()
    else:
        return None


def read_file(folder, prefix, name, dtype=None):
    if name == "node_attributes":
        path = osp.join(folder, '{}_{}.pth'.format(prefix, name))
        return torch.load(path)
    else:
        path = osp.join(folder, '{}_{}.txt'.format(prefix, name))
        if name == "patient_id":
            return read_txt(path)
        else:
            return read_txt_array(path, sep=',', dtype=dtype)


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None


def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    else:
        # Imitate `collate` functionality:
        data._num_nodes = torch.bincount(batch).tolist()
        data.num_nodes = batch.numel()
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    if data.pid is not None:
        slices['pid'] = torch.arange(
            0, batch[-1] + 2, dtype=torch.long)

    return data, slices


def parse_txt_array(src, sep=None, start=0, end=None, dtype=None, device=None):
    to_number = int
    if torch.is_floating_point(torch.empty(0, dtype=dtype)):
        to_number = float

    src = [[to_number(x) for x in line.split(sep)[start:end]] for line in src]
    src = torch.tensor(src).to(dtype).squeeze()
    return src


def read_txt_array(path, sep=None, start=0, end=None, dtype=None, device=None):
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_txt_array(src, sep, start, end, dtype, device)


def read_txt(path):
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return src


def new2old(folder) -> dict:
    """adjust the embedding index (by original graph) to the new index
    (removing empty graph)

    Returns:
        dict: [description]
    """
    index_file = osp.join(folder, 'new2old_index.dict')
    new2old = {}
    with open(index_file, 'r') as f:
        for i, idx in enumerate(f):
            new2old[i] = int(idx.strip())

    return new2old
