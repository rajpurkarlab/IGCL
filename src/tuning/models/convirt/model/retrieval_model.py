"""
Models for retrieval tasks.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from model import vision
from model.pretrain_model import make_projection_layers, BertEncoder
from data.pretrain_loader import get_long_tensor
from utils import constant

logger = logging.getLogger('transfer')

class ImageRetrievalModel(nn.Module):
    """
    A wrapper for a image-image retrieval model. It wraps over an image encoder and
    an image projection layer, which can be loaded from jointly pretrained models.
    """
    def __init__(self, model_name, projection=True, proj_dim=512, proj_layers=2):
        super().__init__()
        self.image_encoder, self.image_dim = getattr(vision, model_name)(pretrained=True)
        # remove classifier from image encoder
        vision.drop_classifier(self.image_encoder)

        # image and text projection layers
        self.image_proj = make_projection_layers(
            self.image_dim,
            proj_dim,
            proj_layers
        ) if projection else None # do not add projection layer for ImageNet weights
    
    def encode(self, imgs, batch_size=64, use_projection=True):
        """
        Encode imgs (B x C x W x H) into hidden vectors (B x d).
        """
        # batchify images
        batches = torch.split(imgs, batch_size, dim=0)
        enc = []
        for b in batches:
            v = self.image_encoder(b) # B x dd
            enc.append(v)
        hidden = torch.cat(enc, dim=0)
        if self.image_proj and use_projection:
            hidden = self.image_proj(hidden)
        return hidden
    
    def score(self, query, candidates, normalize=True):
        """
        Compare a query image with a list of candidate images, and return
        a list of matching scores.
        """
        assert query.size(1) == candidates.size(1)
        if normalize:
            query = F.normalize(query, dim=1) # 1 x dim
            candidates = F.normalize(candidates, dim=1) # B x dim
        # dot product
        scores = torch.mm(query, candidates.transpose(1,0))
        return scores


class TextImageRetrievalModel(nn.Module):
    """
    A wrapper for a text-image retrieval model. It wraps over an image encoder and a text
    encoder, along with two projection layers. The weights are designed to be loaded from
    image-text jointly pretrained models.
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.text_encoder = BertEncoder(opt)
        self.text_dim = self.text_encoder.hidden_size
        self.text_tokenizer = AutoTokenizer.from_pretrained(opt['bert_name'])

        self.image_encoder, self.image_dim = getattr(vision, opt['image_encoder'])(pretrained=True)
        # remove classifier from image encoder
        vision.drop_classifier(self.image_encoder)

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
    
    def encode_image(self, imgs, batch_size=64, use_projection=True):
        """
        Encode imgs (B x C x W x H) into hidden vectors (B x d).
        """
        # batchify images
        batches = torch.split(imgs, batch_size, dim=0)
        enc = []
        for b in batches:
            v = self.image_encoder(b) # B x dd
            enc.append(v)
        hidden = torch.cat(enc, dim=0)
        if use_projection:
            hidden = self.image_proj(hidden)
        return hidden
    
    def encode_text(self, text, use_projection=True):
        """
        Encode a raw input text string into a hidden vector.
        """
        assert isinstance(text, str), "Input text must be a string."
        text_ids = self._tokenize_text(text)
        text_ids = torch.LongTensor(text_ids).cuda().unsqueeze(0) # 1 x L
        text_attention_mask = text_ids.ne(constant.PAD_ID)
        
        hidden = self.text_encoder(text_ids, text_attention_mask)
        if use_projection:
            hidden = self.text_proj(hidden) # 1 x dim
        return hidden

    def _tokenize_text(self, text):
        """
        Accept a single text string.
        """
        tokens = self.text_tokenizer.tokenize(text)
        tokens = [constant.CLS_TOKEN] + tokens + [constant.SEP_TOKEN]
        ids = self.text_tokenizer.convert_tokens_to_ids(tokens)
        return ids
    
    def score(self, text, imgs, normalize=True):
        """
        Compare a query text vector with a list of candidate images, and return
        a list of matching scores.
        """
        assert text.size(1) == imgs.size(1)
        if normalize:
            query = F.normalize(text, dim=1) # 1 x dim
            candidates = F.normalize(imgs, dim=1) # B x dim
        # dot product
        scores = torch.mm(query, candidates.transpose(1,0))
        return scores