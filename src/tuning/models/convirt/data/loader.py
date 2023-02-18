"""
Loaders for different datasets.
"""

import os
import numpy as np
import cv2
import random
import logging
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from utils.helper import get_image_from_dicom
from utils.preprocess import resize_and_normalize, augmentations

CHEXPERT_VALID_COLUMN_NAMES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
CHEXPERT_LABEL_TO_UNCERTAINTY_MAPPING = {'Atelectasis': 1, 'Cardiomegaly': 1, 'Consolidation': 0, 'Edema': 1, 'Pleural Effusion': 1}

logger = logging.getLogger('transfer')

class XRayDataset(Dataset):
    """
    A base XRay image dataset class.
    """
    def __init__(self, meta_file, imsize=224, evaluation=False, augment_p=0):
        raise NotImplementedError("Dataset init function not implemented.")

    def __len__(self):
        return len(self.imgfiles)

    def __getitem__(self, i):
        """
        Get an image, label pair.

        Returns: x, y
            - x: tensorized input
            - y: tensorized label
        """
        X = cv2.imread(self.imgfiles[i])
        # deal with img reading failures: randomly sample another image
        while X is None:
            i = random.randrange(len(self.imgfiles))
            X = cv2.imread(self.imgfiles[i])
        
        # augmentation, resize and normalize
        X = Image.fromarray(X)
        if not self.evaluation and self.augment_p > 0:
            X = augmentations(p=self.augment_p)(X)
        X = resize_and_normalize(imsize=self.imsize)(X)

        y = torch.tensor(self.labels[i])
        return X, y

class RSNADataset(XRayDataset):
    """
    A RSNA pneumonia image dataset, taking a csv file with image path and corresponding label as input.
    """
    def __init__(self, img_dir, csv_file, imsize=224, evaluation=False, augment_p=0, ratio=1.0):
        """
        Args:
            img_dir: root directory of the image files.
            csv_file: filename of the csv meta data file that includes image paths and labels
            imsize: image size after transformation is applied
            evaluation: if True, the data will be used only for evaluation and not training
            augment_p: augmentation probability
            ratio: ratio of samples to use; by default use all.
        """
        self.img_dir = img_dir
        self.csv_file = csv_file
        self.imsize = imsize
        self.evaluation = evaluation
        self.augment_p = augment_p
        self.ratio = ratio

        self.imgfiles, self.labels = self._read_csv()
        N = len(self.imgfiles)
        if ratio < 1.0:
            n_sample = int(N*ratio)
            self.imgfiles = self.imgfiles[:n_sample]
            self.labels = self.labels[:n_sample]
    
    def _read_csv(self):
        path_mapper = lambda x: os.path.join(self.img_dir, x) # remap image paths
        
        # read file and fix values
        df = pd.read_csv(self.csv_file).fillna(0)
        paths = list(map(path_mapper, df['path'].values.tolist()))
        labels = [int(x) for x in df['label'].values.tolist()]

        return paths, labels

class CheXpertDataset(XRayDataset):
    """
    Data loader for the CheXpert dataset.
    """
    def __init__(self, img_dir, csv_file, imsize=224, evaluation=False, augment_p=0, ratio=1.0):
        self.img_dir = img_dir
        self.csv_file = csv_file
        self.imsize = imsize
        self.evaluation = evaluation
        self.augment_p = augment_p
        self.ratio = ratio

        self.imgfiles, self.labels = self._read_csv()
        N = len(self.imgfiles)
        if ratio < 1.0:
            n_sample = int(N*ratio)
            self.imgfiles = self.imgfiles[:n_sample]
            self.labels = self.labels[:n_sample]

        if not self.evaluation:
            self._print_positive_ratios()

    def _read_csv(self):
        path_mapper = lambda x: os.path.join(self.img_dir, x.split('/', 1)[1]) # remap root dir
        
        # read file and fix values
        df = pd.read_csv(self.csv_file).fillna(0.0)
        paths = list(map(path_mapper, df['Path'].values.tolist()))

        # fine-grained uncertainty mapping for labels
        labels_for_columns = []
        for col_name in CHEXPERT_VALID_COLUMN_NAMES:
            col_labels = [float(CHEXPERT_LABEL_TO_UNCERTAINTY_MAPPING[col_name]) if val == -1 else float(val) \
                for val in df[col_name].values.tolist()] # label needs to be float for the BCELoss
            labels_for_columns.append(col_labels)
        labels = list(zip(*labels_for_columns)) # group labels for all columns

        return paths, labels
    
    def _print_positive_ratios(self):
        labels_by_names = list(zip(*self.labels))
        logger.info("-"*30)
        logger.info("Ratio of positive label for each variable:")
        for name, labels in zip(CHEXPERT_VALID_COLUMN_NAMES, labels_by_names):
            pos = int(sum(labels))
            ratio = 100*pos / len(labels)
            logger.info(f"{name}: {pos} ({ratio:.2f}%)")
        logger.info("-"*30)

class COVIDxDataset(XRayDataset):
    """
    Data loader for the COVIDx dataset.
    """
    def __init__(self, img_dir, meta_file, imsize=224, evaluation=False, augment_p=0, ratio=1.0):
        self.img_dir = img_dir
        self.meta_file = meta_file
        self.imsize = imsize
        self.evaluation = evaluation
        self.augment_p = augment_p
        self.ratio = ratio

        self.label2id = {'normal': 0, 'pneumonia': 1, 'covid-19': 2}

        self.imgfiles, self.labels = self._read_meta()
        N = len(self.imgfiles)
        if ratio < 1.0:
            n_sample = int(N*ratio)
            self.imgfiles = self.imgfiles[:n_sample]
            self.labels = self.labels[:n_sample]

        self._print_stats()

    def _read_meta(self):
        """ Load file paths and labels from the meta file.
        """
        labels, paths = [], []
        with open(self.meta_file) as infile:
            for line in infile:
                line = line.strip()
                if len(line) == 0: continue
                fields = line.split()
                assert len(fields) in (3,4), "COVIDx meta file must have 3 or 4 fields."
                paths.append(os.path.join(self.img_dir, fields[1])) # second field is filename
                if fields[2].lower() not in self.label2id: # third field is label
                    raise Exception(f"Unrecognized label: {fields[2]}")
                labels.append(self.label2id[fields[2].lower()])

        return paths, labels
    
    def _print_stats(self):
        logger.info("-"*80)
        total = len(self.imgfiles)
        logger.info(f"Total examples: {total} ({self.ratio*100:g}% of all)")
        for name, idx in self.label2id.items():
            c = sum([1 if l == idx else 0 for l in self.labels])
            ratio = c / total * 100
            logger.info(f"\t{name}: {c} ({ratio:.2f}%)")
        logger.info("-"*80)

class MURADataset(XRayDataset):
    """
    A MURA image dataset, taking a txt file with image paths and a study-level label file as input.
    """
    def __init__(self, img_dir, img_file, label_file, imsize=224, evaluation=False, augment_p=0, ratio=1.0):
        self.img_dir = img_dir
        self.img_file = img_file
        self.label_file = label_file
        self.imsize = imsize
        self.evaluation = evaluation
        self.augment_p = augment_p
        self.ratio = ratio

        self.imgfiles, self.labels, self.study_ids, self.study2label = self._read_csv()
        N = len(self.imgfiles)
        if ratio < 1.0:
            n_sample = int(N*ratio)
            self.imgfiles = self.imgfiles[:n_sample]
            self.labels = self.labels[:n_sample]
            self.study_ids = self.study_ids[:n_sample]
        
        self._print_stats()
    
    def _read_csv(self):
        # read label file
        df = pd.read_csv(self.label_file, names=['study', 'label'])
        study2label = dict()
        for i, row in df.iterrows():
            s = row['study'].rstrip('/')
            l = row['label']
            study2label[s] = int(l)
        
        # read image path file
        files = []
        labels = []
        study_ids = []
        with open(self.img_file) as infile:
            for line in infile:
                line = line.strip()
                if len(line) == 0: continue
                fn = os.path.join(self.img_dir, line)
                sid = '/'.join(line.split('/')[:4])
                files.append(fn)
                study_ids.append(sid)
                if sid not in study2label:
                    raise Exception(f"Cannot find label for study: {sid}")
                labels.append(study2label[sid])

        return files, labels, study_ids, study2label
    
    def _print_stats(self):
        logger.info("-"*80)
        logger.info(f"Total studies: {len(self.study2label)}")
        total = len(self.imgfiles)
        n_pos = sum(self.labels)
        ratio_pos = n_pos / total * 100
        logger.info(f"Kept images: {total} ({self.ratio*100:g}% of all), positive = {n_pos} ({ratio_pos:g}%)")
        logger.info("-"*80)
        

if __name__ == "__main__":
    from utils import logging_config
    # dataset = CheXpertDataset('dataset/chexpert/train.csv')
    # dataset = RSNADataset('dataset/rsna/train.csv')
    # dataset = COVIDxDataset('dataset/COVIDx', 'dataset/COVIDx/train.txt', ratio=0.1)
    dataset = MURADataset('dataset/mura', 'dataset/mura/train.txt', 'dataset/mura/train_labels.csv')
    print(len(dataset))
    labels = dataset.labels
    print(labels[:50])
    print(dataset.imgfiles[:10])
