import torch
from radgraph_dataset import RadGraphDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import *

path = "/deep/group/img-graph"
train_dataset = RadGraphDataset(path, "train").shuffle()
test_dataset = RadGraphDataset(path, "test", use_node_attr=False).shuffle()

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

"""
Inspired by DRACON architecture:
https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/60c74e0f9abda2cf1af8d58a/original/dracon-disconnected-graph-neural-network-for-atom-mapping-in-chemical-reactions.pdf

To be used only with the DRACON architecture.
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SplitConnectedComponents(BaseTransform):
    r"""Removes isolated nodes from the graph."""
    def __call__(self, data):
        num_nodes = data.num_nodes
        undirected_bool = is_undirected(data.edge_index, data.edge_attr)

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

        num_nodes_list = []
        index_list = []
        attr_list = []

        for nodes in cc:

            # Next, get edges for the given sub-graph
            edges = []
            num_edges = data.edge_index.size(dim=1)
            for i in range(num_edges):
                head = data.edge_index[0, i]
                tail = data.edge_index[1, i]
                if head in nodes and tail in nodes:
                    edges.append(i)

            # Obtain edge information for the given graphs
            cc_edge_index = torch.index_select(data.edge_index, 1, torch.tensor(edges).to(device))
            cc_edge_attr = torch.index_select(data.edge_attr, 0, torch.tensor(edges).to(device))

            #Store in lists
            num_nodes_list.append(len(nodes))
            index_list.append(cc_edge_index)
            attr_list.append(cc_edge_attr)

        # Store lists in data structure
        data.edge_index = index_list
        data.edge_attr = attr_list

        # Validate various properties of graph data structure
        assert(len(data.edge_index) == len(cc))
        assert(len(data.edge_attr) == len(cc))

        for i in range(len(cc)):
            assert(data.edge_index[i].dim() == 2)
            assert(data.edge_attr[i].dim() == 2)
            assert(data.edge_index[i].size(dim=0) == 2)
            assert(data.edge_index[i].size(dim=1) == data.edge_attr[i].size(dim=0))
            if num_nodes_list[i] > 1:
                assert(not contains_isolated_nodes(data.edge_index[i], num_nodes_list[i]))

        return data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def test(loader):

    for data in loader:
        data = data.to(device)
        data = SplitConnectedComponents()(data)
    print("Passed transform assertions.")

if __name__ == "__main__":
    test(test_loader)