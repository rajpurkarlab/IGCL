"""
Loaders for different datasets.
"""
import os
import numpy as np
import cv2
import logging
import random
import json
from collections import OrderedDict
from copy import deepcopy
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import constant
from utils.helper import get_image_from_dicom
from utils.preprocess import pretrain_augmentations, resize_and_normalize

logger = logging.getLogger('transfer')

def load_json(filename):
    with open(filename) as infile:
        data = json.load(infile)
    return data

class PretrainDataset(Dataset):
    """
    A dataset for handling data for pretraining.

    Args:
        - indexed_file: a json file with two fields:
            `sentences` for all individual sentences; and 
            `indexed_reports` that maps a study id to a list of sentence indices
        - meta_file: a meta data json file which maps a study id to a list of [id, pid, view]
    """
    def __init__(self, indexed_file, meta_file, img_dir, opt, tokenizer, evaluation=False, imsize=224, augment_p=0.95):
        self.indexed_file = indexed_file
        self.meta_file = meta_file
        self.img_dir = img_dir
        self.opt = opt
        self.tokenizer = tokenizer
        self.evaluation = evaluation
        self.imsize = imsize
        self.augment_p = augment_p

        self.all_sentences, self.indexed_reports, self.meta_data = self._load_data()
        # resolve image paths and tokenize sentences
        self._prepare_data()
    
    def _load_data(self):
        """
        Load from the indexed_file and the meta_file.
        """
        # read from files
        report_data = load_json(self.indexed_file)
        all_sentences = report_data['sentences']
        indexed_reports = report_data['indexed_reports']
        meta_data = load_json(self.meta_file)
        return all_sentences, indexed_reports, meta_data
    
    def _prepare_data(self):
        """
        Prepare the data by resolving paths for image files, and finding and tokenizing sentences.
        """
        # each image group is a list of paths for images corresponding to that study
        # each text group is a list of sentences corresponding to that study
        self.image_groups, self.text_groups = [], []
        
        def resolve_image_paths(study_id):
            """ Resolve the path of an image given an ID. """
            # full study id has a "s" at the beginning, while the meta does not have
            if study_id[1:] not in self.meta_data:
                raise Exception(f"Cannot find study id from meta data: {study_id[1:]}")
            metas = self.meta_data[study_id[1:]]
            paths = []
            for m in metas:
                iid, pid = m['id'], 'p{}'.format(m['pid']) # actual pid needs to start with p
                assert len(pid) > 3
                # hardcoded pattern for image path
                img_path = os.path.join(self.img_dir, pid[:3], pid, study_id, f"{iid}.png")
                paths.append(img_path)
            return paths
        
        def encode_text(text):
            """ Tokenize and encode the text into ids. """
            # if self.opt['lower']:
                # text = text.lower()
            tokens = self.tokenizer.tokenize(text)
            # for BERT encoding
            tokens = [constant.CLS_TOKEN] + tokens + [constant.SEP_TOKEN]
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            return ids

        # iterate through studies in self.indexed_reports
        for sid, sentences in self.indexed_reports.items():
            # TODO: use all sections for now; more filters can be added here
            # resolve and tokenize sentences
            text_group = []
            for sent_id in sentences:
                text = self.all_sentences[sent_id]
                if len(text.split()) <= 3: # remove sentences with <=3 words
                    continue
                text_group.append(encode_text(text))
            if len(text_group) == 0: # skip studies with no sentences
                continue
            self.text_groups.append(text_group)

            # find images
            image_group = resolve_image_paths(sid)
            self.image_groups.append(image_group)
        
        assert len(self.image_groups) == len(self.text_groups), \
            "Numbers of image groups and text groups mismatch."
        
        avg_img = sum([len(g) for g in self.image_groups]) / len(self.image_groups)
        avg_text = sum([len(g) for g in self.text_groups]) / len(self.text_groups)
        logger.info(f"{len(self.image_groups)} examples loaded.")
        logger.info(f"On average: # {avg_img} images, # {avg_text} sentences per study.")
        return

    def __len__(self):
        return len(self.image_groups)
    
    def __getitem__(self, i):
        """
        For the i-th study, return two data points:
            - a sampled image view, which is from a sampled image
            - a sampled sentence from the sentence groups
        """
        # sample an image; if evaluation, use first image
        mode = cv2.IMREAD_GRAYSCALE
        img_path = random.choice(self.image_groups[i]) if not self.evaluation else self.image_groups[i][0]
        X = cv2.imread(img_path, mode)
        # deal with img reading failures: randomly sample another image
        while X is None:
            # change i to a random index
            i = np.random.choice(len(self.image_groups))
            img_path = random.choice(self.image_groups[i]) if not self.evaluation else self.image_groups[i][0]
            X = cv2.imread(img_path, mode)
        X = np.repeat(np.expand_dims(X, axis=-1), 3, axis=-1)

        # sample an image view by augmentation, resize and normalize
        X = Image.fromarray(X)
        if not self.evaluation:
            X = pretrain_augmentations(imsize=self.imsize, p=self.augment_p)(X)
        img = resize_and_normalize(imsize=self.imsize)(X)
        
        # sample a positive sentence
        text = random.choice(self.text_groups[i]) if not self.evaluation else self.text_groups[i][0]
        return img, text

class RIHPretrainDataset(PretrainDataset):
    """
    A dataset for handling RIH data for NCE-based pretraining.
    """
    
    def _prepare_data(self):
        """
        Prepare the data by resolving paths for image files, and finding and tokenizing sentences.
        """
        # each image group is a list of paths for images corresponding to that study
        # each text group is a list of sentences corresponding to that study
        self.image_groups, self.text_groups = [], []
        
        def resolve_image_paths(study_id):
            """ Resolve the path of an image given an ID. """
            # look up study id
            if study_id not in self.meta_data:
                raise Exception(f"Cannot find study id from meta data: {study_id}")
            metas = self.meta_data[study_id]
            paths = []
            for m in metas:
                img_path = os.path.join(self.img_dir, m['file'])
                paths.append(img_path)
            return paths
        
        def encode_text(text):
            """ Tokenize and encode the text into ids. """
            # if self.opt['lower']:
                # text = text.lower()
            tokens = self.tokenizer.tokenize(text)
            # for BERT encoding
            tokens = [constant.CLS_TOKEN] + tokens + [constant.SEP_TOKEN]
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            return ids

        # iterate through studies in self.indexed_reports
        for sid, sentences in self.indexed_reports.items():
            # TODO: use all sections for now; more filters can be added here
            # resolve and tokenize sentences
            text_group = []
            for sent_id in sentences:
                text = self.all_sentences[sent_id]
                if len(text.split()) <= 3: # remove sentences with <=3 words
                    continue
                text_group.append(encode_text(text))
            if len(text_group) == 0: # skip studies with no sentences
                continue
            self.text_groups.append(text_group)

            # find images
            image_group = resolve_image_paths(sid)
            self.image_groups.append(image_group)
        
        assert len(self.image_groups) == len(self.text_groups), \
            "Numbers of image groups and text groups mismatch."
        
        avg_img = sum([len(g) for g in self.image_groups]) / len(self.image_groups)
        avg_text = sum([len(g) for g in self.text_groups]) / len(self.text_groups)
        logger.info(f"{len(self.image_groups)} examples loaded.")
        logger.info(f"On average: # {avg_img} images, # {avg_text} sentences per study.")
        return


class NCEPretrainDataLoader(DataLoader):
    """ A data loader for the pretraining dataset.  """
    def __init__(self, dataset, opt, **kwargs):
        if kwargs.get('collate_fn', None) is None:
            kwargs['collate_fn'] = self._collate_fn

        self.opt = opt
        super().__init__(dataset, **kwargs)

    def _collate_fn(self, batch_data):
        """
        Collate a batch of data into tensors.
        """
        batch_size = len(batch_data)
        batch = list(zip(*batch_data))
        assert len(batch) == 2

        img = torch.stack(batch[0], dim=0) # stack all images
        text_ids = get_long_tensor(batch[1], batch_size)
        text_attention_mask = text_ids.ne(constant.PAD_ID)

        tensorized = OrderedDict()
        tensorized['image'] = [img,]
        tensorized['text'] = [text_ids, text_attention_mask]
        return tensorized

class BinaryPretrainDataLoader(DataLoader):
    """ A data loader for the pretraining dataset.  """
    def __init__(self, dataset, opt, **kwargs):
        if kwargs.get('collate_fn', None) is None:
            kwargs['collate_fn'] = self._collate_fn

        self.opt = opt
        super().__init__(dataset, **kwargs)

    def _collate_fn(self, batch_data):
        """
        Collate a batch of data into tensors.
        """
        batch_size = len(batch_data)
        batch = list(zip(*batch_data))
        assert len(batch) == 2
        
        img = batch[0]
        text = batch[1]
        # create positive and negative text examples
        text_pos = list(text)
        text_neg = deepcopy(text_pos)
        random.shuffle(text_neg)
        labels = [1] * batch_size + [0] * batch_size
        batch_size = batch_size * 2 # actual batch_size is doubled

        # create tensor
        img = torch.stack(batch[0], dim=0) # stack all images
        img = torch.cat([img, img], dim=0) # repeat images for negative chunk
        text_ids = text_pos + text_neg
        text_ids = get_long_tensor(text_ids, batch_size)
        text_attention_mask = text_ids.ne(constant.PAD_ID)
        labels = torch.LongTensor(labels)

        tensorized = OrderedDict()
        tensorized['image'] = [img,]
        tensorized['text'] = [text_ids, text_attention_mask]
        tensorized['label'] = [labels,]
        return tensorized

def get_long_tensor(tokens_list, batch_size, pad=constant.PAD_ID):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(pad)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens
