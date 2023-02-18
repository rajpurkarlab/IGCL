import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import *

class AddDummyNode(BaseTransform):
    r"""Removes isolated nodes from the graph."""
    def __call__(self, data):
        # Increase dimension of edge_attr by 1 (for relations where at least one node is a dummy node)
        edge_dim = data.edge_attr.shape[1]
        num_edges = data.edge_attr.shape[0]
        zero_cols = torch.zeros((num_edges, 1)).to(data.edge_attr.device)
        data.edge_attr = torch.cat((data.edge_attr, zero_cols), dim=1)
        
        # print("DATA in AddDummyNode", data)
        num_nodes = data.num_nodes
        # undirected_bool = is_undirected(data.edge_index, data.edge_attr)
        # print("UNDIRECTED_BOOL", undirected_bool)
        if not contains_isolated_nodes(data.edge_index, data.num_nodes):
            return data

        # Add dummy node to data.
        new_row = torch.zeros(1, data.x.size(dim=1)).to(data.x.device)
        data.x = torch.cat((data.x, new_row), dim=0)
        
        dummy_index = num_nodes

        # Connect every node to dummy node
        for node_index in range(0, dummy_index):
            # Add connection
            connection = torch.tensor([[node_index], [dummy_index]]).to(data.edge_index.device)
            # print(data.pid, 'connection1', connection.shape)
            # print(data.pid, 'edge_index1', data.edge_index.shape, data.edge_index.shape[1], data.edge_index)
            if data.edge_index.shape[1] == 0:  # empty edge_index
                # print("EMPTY EDGE INDEX")
                data.edge_index = connection
            else:
                data.edge_index = torch.cat((data.edge_index, connection), dim=1)

            # Add attribute
            new_row = torch.zeros(1, data.edge_attr.size(dim=1)).to(data.edge_attr.device)
            new_row[0][-1] = 1.
            data.edge_attr = torch.cat((data.edge_attr, new_row), dim=0)

            # Add reverse connection if undirected. Actually just add reverse connection always.
            # if undirected_bool:
                # Add connection
            connection = torch.tensor([[dummy_index], [node_index]]).to(data.edge_index.device)

            # print(data.pid, 'connection2', connection.shape)
            # print(data.pid, 'edge_index2', data.edge_index.shape, data.edge_index.shape[1], data.edge_index)

            if data.edge_index.shape[1] == 0:  # empty edge_index
                data.edge_index = connection
            else:
                data.edge_index = torch.cat((data.edge_index, connection), dim=1)

            # Add attribute
            new_row = torch.zeros(1, data.edge_attr.size(dim=1)).to(data.edge_attr.device)
            new_row[0][-1] = 1.
            data.edge_attr = torch.cat((data.edge_attr, new_row), dim=0)

        # Graph should not have isolated parts now.
        assert(not contains_isolated_nodes(data.edge_index, data.num_nodes))

        return data