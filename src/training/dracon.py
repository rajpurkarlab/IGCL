import torch
from torch import nn
from torch.nn import MultiheadAttention, Linear, Dropout
from torch.nn.modules.transformer import LayerNorm
from torch_geometric.nn import RGCNConv, GlobalAttention
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse
from torch_geometric.nn import GlobalAttention, global_mean_pool

# Dataset needs to include adjacency matrix

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)

    def forward(self, features):
        return self.embed(features)

class RGCNModel(torch.nn.Module):
    def __init__(self, in_dim, h_dim = 160, num_rels = 3, num_layers = 6, activation = F.relu):
        super(RGCNModel, self).__init__()
        layers = []
        layers.append(RGCNConv(in_dim, h_dim, num_rels))
        for _ in range(num_layers - 1):
            layers.append(RGCNConv(h_dim, h_dim, num_rels))
        self.layers = nn.ModuleList(layers)
        self.activation = activation

    def forward(self, x, edge_index, edge_type):
        for index, layer in enumerate(self.layers):
            x = self.activation(layer(x, edge_index, edge_type))
        return x

class TransLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super(TransLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=1024, dropout=0.1):
        super(TransModel, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(TransLayer(d_model, nhead))
        self.layers = nn.ModuleList(layers)

    def forward(self, src):
        h = src
        for layer in self.layers:
            h = layer(h)
        return h

class FCNModel(nn.Module):
    def __init__(self, d_model, num_layers=3):
        super(FCNModel, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.ReLU())
        # layers.append(nn.Linear(d_model, 1))  # Why would we have this linear layer?
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DRACON(nn.Module):
    def __init__(self,
                 feat_size,
                 h_dim=64,
                 num_rels=3,
                 num_conv_layers=6,
                 num_trans_layers=1,
                 num_fcn_layers=1,
                 num_attention_heads=1,
                 use_pool=True):

        super(DRACON, self).__init__()
        self.h_dim = h_dim
        self.rgcn = RGCNModel(feat_size, self.h_dim, num_rels, num_conv_layers)
        self.trans = TransModel(self.h_dim, num_attention_heads, num_trans_layers)
        self.fcn = FCNModel(self.h_dim, num_layers=num_fcn_layers)
        self.use_pool = use_pool
        if not self.use_pool:
            self.att = GlobalAttention(nn.Linear(self.h_dim, 1))

    def to_sparse_batch(self, x, adj, mask=None):

        B, N_max, D = x.shape
        # get num of nodes and reshape x
        num_nodes_graphs = torch.zeros_like(x[:,0,0], dtype=torch.int64).fill_(N_max)
        x = x.reshape(-1, D) # total_nodes * D

        # apply mask 
        if mask is not None:
            # mask adj
            adj = (adj * mask.unsqueeze(2)).transpose(1,2) 
            adj = (adj * mask.unsqueeze(2)).transpose(1,2) 
            # get number nodes per graph 
            num_nodes_graphs = mask.sum(dim=1)  # B
            # mask x
            x = x[mask.reshape(-1)] # total_nodes * D

        return x

    def forward(self, data):
        x, edge_index, edge_attr, batch, num_graphs = data.x, data.edge_index, data.edge_attr, data.batch, data.num_graphs

        # Convert from one-hot encoding to label encoding
        edge_type = torch.argmax(edge_attr, dim=1).to(data.x.device)

        #print(f"0. {x.shape}")

        x = self.rgcn(data.x, edge_index, edge_type)
        #print(f"1. {x.shape}")

        x, mask = to_dense_batch(x, batch) # (batch_size, num_nodes, h_dim)
        adj = to_dense_adj(edge_index, batch=batch)
        #print(f"2. {x.shape}")
        #print(f"Mask: {mask.shape}")
        #print(f"Adj: {adj.shape}")

        if self.trans is not None:
            x = self.trans(x.permute(1, 0, 2)).permute(1, 0, 2) # (batch_size, num_nodes, h_dim)
        #print(f"3. {x.shape}")

        x = self.to_sparse_batch(x, adj, mask)
        #print(f"4. {x.shape}")

        if self.use_pool:
            x = global_mean_pool(x, batch)
        else:
            x = self.att(x, batch)

        #print(f"5. {x.shape}")

        x = self.fcn(x)

        #print(f"6. {x.shape}")

        return x

