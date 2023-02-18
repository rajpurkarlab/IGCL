import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import *

"""
Inspired by DRACON architecture:
https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/60c74e0f9abda2cf1af8d58a/original/dracon-disconnected-graph-neural-network-for-atom-mapping-in-chemical-reactions.pdf
"""

class AddReactionNodes(BaseTransform):
    r"""Removes isolated nodes from the graph."""
    def __call__(self, data):
        # Increase dimension of edge_attr by 1 (for relations where at least one node is a dummy node)
        edge_dim = data.edge_attr.shape[1]
        num_edges = data.edge_attr.shape[0]
        zero_cols = torch.zeros((num_edges, 1)).to(data.edge_attr.device)
        data.edge_attr = torch.cat((data.edge_attr, zero_cols), dim=1)
        
        num_nodes = data.num_nodes
        undirected_bool = is_undirected(data.edge_index, data.edge_attr)
        if not contains_isolated_nodes(data.edge_index, data.num_nodes):
            return data

        # Build adjcency data structure from edge_index
        adj = [[] for i in range(num_nodes)]
        num_edges = data.edge_index.size(dim=1)
        for i in range(num_edges):
            head = data.edge_index[0, i]
            tail = data.edge_index[1, i]
            adj[head].append(tail)
            adj[tail].append(head)
        # Ensure duplicates are not included
        adj = [list(set(edge_lst)) for edge_lst in adj]

        def dfs(temp, cur_node, visited):
            # Mark the current vertex as visited
            visited[cur_node] = True
     
            # Store the vertex to list
            temp.append(cur_node)
     
            # Repeat for all vertices adjacent
            # to this vertex v
            for i in adj[cur_node]:
                if visited[i] == False:
                    # Update the list
                    temp = dfs(temp, i, visited)
            return temp

        def connected_components():
            visited = []
            cc = []
            for i in range(num_nodes):
                visited.append(False)
            for cur_node in range(num_nodes):
                if visited[cur_node] == False:
                    temp = []
                    cc.append(dfs(temp, cur_node, visited))
            return cc

        cc = connected_components()
        if len(cc) == 1:
            # Only one component; return
            return data

        for i in range(len(cc)):
            # Add dummy node for each connected component
            new_row = torch.zeros(1, data.x.size(dim=1)).to(data.x.device)
            data.x = torch.cat((data.x, new_row), dim=0)

        first_dummy_index = num_nodes

        def make_connection(node_a, node_b):
            # Add connection
            connection = torch.tensor([[node_a], [node_b]]).to(data.edge_index.device)
            if data.edge_index.shape[1] == 0:  # empty edge_index
                data.edge_index = connection
            else:
                data.edge_index = torch.cat((data.edge_index, connection), dim=1)

            # Add attribute
            new_row = torch.zeros(1, data.edge_attr.size(dim=1)).to(data.edge_attr.device)
            new_row[0][-1] = 1.
            # if data.edge_attr.shape[1] == 0:  # empty edge_attr
            #     data.edge_attr = new_row
            # else:
            data.edge_attr = torch.cat((data.edge_attr, new_row), dim=0)

            # Add reverse connection if undirected
            # if undirected_bool:
            # Add connection
            connection = torch.tensor([[node_b], [node_a]]).to(data.edge_index.device)
            data.edge_index = torch.cat((data.edge_index, connection), dim=1)

            # Add attribute
            new_row = torch.zeros(1, data.edge_attr.size(dim=1)).to(data.edge_attr.device)
            new_row[0][-1] = 1.
            data.edge_attr = torch.cat((data.edge_attr, new_row), dim=0)

        # Connect dummy node to its connected component
        for index, nodes in enumerate(cc):
            dummy_index = first_dummy_index + index 
            for node in nodes:
                make_connection(node, dummy_index)

        # Create reaction node
        reaction_row = torch.zeros(1, data.x.size(dim=1)).to(data.x.device)
        data.x = torch.cat((data.x, reaction_row), dim=0)

        # Densely connect dummy nodes to the reaction node
        reaction_index = first_dummy_index + len(cc)
        for index in range(len(cc)):
            dummy_index = first_dummy_index + index
            make_connection(reaction_index, dummy_index)

        # Graph should not have isolated parts now.
        assert(not contains_isolated_nodes(data.edge_index, data.num_nodes))

        return data