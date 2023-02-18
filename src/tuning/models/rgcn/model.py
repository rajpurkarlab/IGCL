from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool, GAT, GlobalAttention, SAGEConv, RGCNConv
from transformers import CLIPProcessor, CLIPVisionModel


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)

    def forward(self, x: torch.Tensor):
        # print('FORWARD IN VISUALTRANSFORMER')
        # print(x.shape)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])
        
        # print("VISUALTRANSFORMER OUTPUT SHAPE", x.shape)

        return x


# class BERTEncoder(nn.Module):
#     """"Just a linear projection layer on top of frozen BERT embeddings"""
#     def __init__(self, bert_dim, output_dim):
#         super().__init__()
        
#         # bert_dim is 512 for 'openai/clip-vit-base-patch32'
#         scale = bert_dim ** -0.5
#         self.proj = torch.nn.Parameter(scale * torch.randn(bert_dim, output_dim))
#         # nn.init.normal_(self.proj, std=bert_dim ** -0.5)

#     def forward(self, bert_embeddings):
#         return bert_embeddings @ self.proj

class MyGCN(torch.nn.Module):
    def __init__(self, num_features: int, num_layers: int, hidden: int):
        # print("GCN", num_features, num_layers, hidden, type(num_features), type(num_layers), type(hidden))
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = nn.Linear(hidden, hidden)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        return x

    def __repr__(self):
        return self.__class__.__name__
    

class MyGAT(torch.nn.Module):
    def __init__(self, num_features: int, num_layers: int, hidden: int, edge_dim: int):
        # print("GCN", num_features, num_layers, hidden, type(num_features), type(num_layers), type(hidden))
        super().__init__()
        self.gat = GAT(in_channels=num_features, hidden_channels=hidden, num_layers=num_layers, edge_dim=edge_dim)
        self.att = GlobalAttention(nn.Linear(hidden, 1))
        self.lin1 = nn.Linear(hidden, hidden)

    def reset_parameters(self):
        self.gat.reset_parameters()
        self.att.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        out = self.gat(x, edge_index, edge_attr)
        print("1", out.shape)
        out = self.att(out, batch)
        print("2", out.shape)
        out = self.lin1(out)
        print("3", out.shape)
        return out

    def __repr__(self):
        return self.__class__.__name__
    
class NoEdgeAttrGAT(torch.nn.Module):
    def __init__(self, num_features: int, num_layers: int, hidden: int):
        super().__init__()
        self.gat = GAT(in_channels=num_features, hidden_channels=hidden, num_layers=num_layers)
        self.att = GlobalAttention(nn.Linear(hidden, 1))
        self.lin1 = nn.Linear(hidden, hidden)

    def reset_parameters(self):
        self.gat.reset_parameters()
        self.att.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        out = self.gat(x, edge_index)
        out = self.att(out, batch)
        out = self.lin1(out)
        return out

    def __repr__(self):
        return self.__class__.__name__
    

# Written by Sameer
class GlobalAttentionNet(torch.nn.Module):
    def __init__(self, num_features: int, num_layers: int, hidden: int):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden)
        self.convs = nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.att = GlobalAttention(nn.Linear(hidden, 1))
        self.lin1 = nn.Linear(hidden, hidden)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.att.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.att(x, batch)
        x = self.lin1(x)
        return x

    def __repr__(self):
        return self.__class__.__name__
    
    
class MyRGCN(torch.nn.Module):
    def __init__(self, num_features: int, num_layers: int, hidden: int, edge_dim: int):
        super().__init__()
        self.conv1 = RGCNConv(num_features, hidden, edge_dim)
        self.convs = nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden, hidden, edge_dim))
        self.lin1 = nn.Linear(hidden, hidden)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # Convert from one-hot encoding to label encoding
        edge_attr = edge_attr.cpu()
        edge_type = torch.tensor(np.where(edge_attr==1)[1]).to(x.device)
        
        x = F.relu(self.conv1(x, edge_index, edge_type))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_type))
        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        return x

    def __repr__(self):
        return self.__class__.__name__
    
    
class HuggingFaceImageEncoder(nn.Module):
    """"Wrapper for HuggingFace pretrained CLIP image encoder"""
    def __init__(self, device, model_name='openai/clip-vit-base-patch32'):
        super().__init__()
        
        # bert_dim is 512 for 'openai/clip-vit-base-patch32'
        self.model = CLIPVisionModel.from_pretrained(model_name)
        # self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = device
        
        # Freeze HuggingFace model as a test
        # print("FREEZING HUGGING FACE IMAGE ENCODER")
        # for param in self.parameters():
        #     param.requires_grad = False

    def forward(self, image):
        # inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        # outputs = self.model(**inputs)
        outputs = self.model(pixel_values=image)
        return outputs.pooler_output
       
class Encoder(nn.Module):
    """"Just a linear projection layer on top of some encoder"""
    def __init__(self, encoder, encoder_dim, output_dim=None):
        super().__init__()
        
        # bert_dim is 512 for 'openai/clip-vit-base-patch32'
        scale = encoder_dim ** -0.5
        self.encoder = encoder
        if output_dim is not None:
            self.proj = torch.nn.Parameter(scale * torch.randn(encoder_dim, output_dim))

    def forward(self, data):
        output = self.encoder(data)
        if self.proj is not None:
            output = output @ self.proj
        return output

    
class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_encoder,
                 image_encoder_dim: int,
                 # graph
                 graph_encoder,
                 graph_encoder_dim: int,
                 ):
        super().__init__()

        self.visual = Encoder(image_encoder, image_encoder_dim, embed_dim)
        self.gnn = Encoder(graph_encoder, graph_encoder_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @property
    def dtype(self):
        return self.visual.proj.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_graph(self, graph):
        x = self.gnn(graph)
        return x

    def forward(self, image, graph):
        # print('IMAGE TYPE', image.shape, image)
        image_features = self.encode_image(image)
        # print('GRAPH TYPE', graph.shape, graph)
        graph_features = self.encode_graph(graph)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        graph_features = graph_features / graph_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ graph_features.t()
        logits_per_graph = logit_scale * graph_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_graph
    
    
class FinetunedModel(torch.nn.Module):
    def __init__(self, clip: CLIP, nclass: int):
        """
        clip: CLIP model with image encoder
        nclass: number of possible classes for output (14 for CheXpert)
        """
        super(FinetunedModel, self).__init__()
        
        self.clip = clip  # don't freeze clip, so entire model is trained
        # print(clip.dtype)
        embedding_dim = clip.visual.proj.shape[1]  # 768
        self.linear_head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, nclass)
        )
        # self.linear_head = torch.nn.Linear(embedding_dim, nclass, dtype=torch.half)
        # scale = embedding_dim ** -0.5
        # self.linear_head = torch.nn.Parameter(scale * torch.randn(embedding_dim, nclass, dtype=torch.half))
        # torch.nn.init.normal_(self.linear_head, std=scale)

    def forward(self, image):
        embedding = self.clip.encode_image(image)
        # print("EMBEDDING SHAPE", embedding.shape, embedding.type())
        # print(embedding)
        output = self.linear_head(embedding.float())
        # output = embedding @ self.linear_head
        return output

    
class LinearModel(torch.nn.Module):
    def __init__(self, clip: CLIP, nclass: int):
        """
        clip: CLIP model with image encoder
        nclass: number of possible classes for output (14 for CheXpert)
        """
        super(LinearModel, self).__init__()
        
        self.clip = clip
        # Freeze CLIP model, so only linear layer is trained
        for param in self.clip.parameters():
            param.requires_grad = False
    
        # embedding_dim = clip.visual.proj.shape[0]  # 768
        embedding_dim = clip.visual.proj.shape[1]  # 512
        # print("EMBEDDING DIM", embedding_dim)
        self.linear_head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, nclass)
        )

    def forward(self, image):
        # print("IMAGE IN LINEAR MODEL", image)
        embedding = self.clip.encode_image(image)
        # print("EMBEDDING SHAPE", embedding.shape)
        # print(embedding)
        output = self.linear_head(embedding.float())
        return output
