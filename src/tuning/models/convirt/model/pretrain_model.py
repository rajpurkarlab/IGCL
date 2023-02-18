"""
Model architectures for joint pretraining.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from model import vision
from utils import constant

logger = logging.getLogger('transfer')

class BertEncoder(nn.Module):
    """
    A sentence encoder based on BERT.
    """
    def __init__(self, opt, bert_model=None):
        super().__init__()
        self.opt = opt
        if bert_model is not None:
            self.bert_model = bert_model
        else: # load with model name
            self.bert_model = AutoModel.from_pretrained(opt['bert_name'])
        
        self.hidden_size = self.bert_model.config.hidden_size
        self.num_layers = len(self.bert_model.encoder.layer)
    
    def forward(self, ids, attention_mask):
        
        outputs = self.bert_model(ids, attention_mask=attention_mask)

        if self.opt['pool'] == 'cls':
            pooled = outputs[1]
        else: # use other pooling
            pool_mask = attention_mask.eq(0).unsqueeze(2) # invert the mask for pooling
            pooled = pool(outputs[0], pool_mask, self.opt['pool'])

        # batch_size x hidden_size
        return pooled
    
    def freeze_first_n_layers(self, n_first_layers, freeze_embeddings=True):
        """
        Freeze the first N layers of the BERT encoder, and do not finetune them during backpropagation.
        """
        if freeze_embeddings:
            for param in list(self.bert_model.embeddings.parameters()):
                param.requires_grad = False
            logger.info("Freezing embedding layer of BERT model.")
        if n_first_layers < 0:
            n_first_layers = self.num_layers
        for idx in range(n_first_layers):
            for param in list(self.bert_model.encoder.layer[idx].parameters()):
                param.requires_grad = False
        logger.info(f"Freezing first {n_first_layers} layers of the BERT Encoder.")


class JointBinaryModel(nn.Module):
    """
    A joint image and text classifier module that predicts the contrastive label.
    """
    def __init__(self, opt, bert_model=None):
        super().__init__()
        self.opt = opt
        self.num_class = 2

        self.text_encoder = BertEncoder(opt, bert_model)
        self.text_dim = self.text_encoder.hidden_size
        self.image_encoder, self.image_dim = getattr(vision, opt['image_encoder'])(pretrained=True)
        # remove classifier from image encoder
        vision.drop_classifier(self.image_encoder)

        self.dropout = nn.Dropout(self.opt['dropout'])

        self.num_clf_layer = self.opt['num_clf_layer']
        if self.num_clf_layer == 1:
            self.classifier = nn.Linear(self.text_dim + self.image_dim, self.num_class)
        elif self.num_clf_layer > 1:
            self.hidden_dim = self.opt['clf_hidden_dim']
            in_dim = self.text_dim + self.image_dim
            layers = []
            for i in range(self.num_clf_layer-1):
                layers += [
                    nn.Linear(in_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.opt['dropout'])
                ]
                in_dim = self.hidden_dim
            layers.append(nn.Linear(in_dim, self.num_class))
            self.classifier = nn.Sequential(*layers)
        else:
            raise Exception(f"Invalid number of classifier layers: {self.num_clf_layer}")

        # freeze some layers in the network
        if self.opt['freeze_text_encoder']:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
    
    def forward(self, image, text_ids, text_attention_mask):
        img_v = self.image_encoder(image)
        
        ctx = torch.enable_grad
        if self.opt['freeze_text_encoder']:
            ctx = torch.no_grad
        with ctx():
            text_v = self.text_encoder(text_ids, text_attention_mask)

        v = torch.cat([img_v, text_v], dim=-1)
        v = self.dropout(v)
        logits = self.classifier(v)
        return logits


class JointNCEModel(nn.Module):
    """
    An NCE-based joint image and text representation learner.
    """
    def __init__(self, opt, bert_model=None):
        super().__init__()
        self.opt = opt

        self.text_encoder = BertEncoder(opt, bert_model)
        self.text_dim = self.text_encoder.hidden_size
        self.image_encoder, self.image_dim = getattr(vision, opt['image_encoder'])(pretrained=True)
        # remove classifier from image encoder
        vision.drop_classifier(self.image_encoder)

        dropout_p = self.opt['dropout']
        self.dropout = nn.Dropout(dropout_p)

        # image and text projection layers
        self.image_proj = make_projection_layers(
            self.image_dim,
            self.opt['proj_dim'],
            self.opt['proj_layers']
        )
        self.text_proj = make_projection_layers(
            self.text_dim,
            self.opt['proj_dim'],
            self.opt['proj_layers']
        )

        # freeze some layers in the network
        self.text_encoder.freeze_first_n_layers(
            self.opt['freeze_text_first_n_layers'],
            self.opt['freeze_text_embeddings']
        )

        # temperatures
        self.t1 = self.opt.get('t1', 0.1)
        self.t2 = self.opt.get('t2', 0.1)
        self.biloss = self.opt.get('biloss', False)
    
    def forward(self, image, text_ids, text_attention_mask):
        img_v = self.image_encoder(image)
        text_v = self.text_encoder(text_ids, text_attention_mask)

        img_v = self.image_proj(img_v) # batch_size, dim
        text_v = self.text_proj(text_v) # batch_size, dim

        # normalize for cosine similarity
        img_v = F.normalize(img_v, dim=1)
        text_v = F.normalize(text_v, dim=1)
        
        # dot product with temperature
        logits = torch.mm(img_v, text_v.transpose(1,0))
        logits /= self.t1

        if self.biloss:
            logits2 = torch.mm(text_v, img_v.transpose(1,0))
            logits2 /= self.t2
        else:
            logits2 = None

        return logits, logits2

def make_projection_layers(in_dim, out_dim, num_layers=2, dropout_layer=None):
    if num_layers == 1: # simple linear layer
        return nn.Linear(in_dim, out_dim)
    else:
        layers = []
        for i in range(num_layers-1):
            layers += [
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
            ]
            if dropout_layer:
                layers += [dropout_layer]
        layers += [nn.Linear(in_dim, out_dim)]
        return nn.Sequential(*layers)

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'mean':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    elif type == 'sum':
        h = h.masked_fill(mask, 0)
        return h.sum(1)
    else:
        raise Exception("Unsupported pooling type: " + type)