import cv2
import torch
import pandas as pd
from PIL import Image

import util
from .base_dataset import BaseDataset
from constants import *


class RANZCRDataset(BaseDataset):
    def __init__(self, csv_name, is_training, study_level,
                 transform_args, toy, return_info_dict, logger=None, data_args=None):
        # Pass in parent of data_dir because test set is in a different
        # directory due to dataset release, and uncertain maps are in a
        # different directory as well (both are under the parent directory).
        super().__init__(csv_name, is_training, transform_args)
        self.study_level = study_level
        self.toy = toy
        self.return_info_dict = return_info_dict
        self.logger = logger
        self.data_args = data_args

        self.is_train_dataset = self.csv_name == "train.csv"
        self.is_val_dataset = self.csv_name == "valid.csv"
        self.is_test_dataset = self.csv_name == "test.csv"
        
        if self.is_test_dataset:
            self.csv_path  = RANZCR_DATA_DIR / self.csv_name  # Modify this
        else:
            self.csv_path =  RANZCR_DATA_DIR / self.csv_name

        if self.is_val_dataset:
            print("valid", self.csv_path)
        df = pd.read_csv(Path(self.csv_path))

        self.labels = self.get_labels(df)
        self.img_paths = self.get_paths(df)

    def set_study_as_index(self, df):
        df.index = df[COL_STUDY]

    def get_paths(self, df):
        return df[RANZCR_COLPATH]

    def get_validationImages(self,index):
        label = self.labels.iloc[index].values 
        label = torch.FloatTensor(label)

        img_paths = pd.Series(self.img_paths.iloc[index])
        imgs = [Image.open(path).convert('RGB') for path in img_paths]

        imgs = [self.transform(img) for img in imgs]
        imgs = torch.stack(imgs)

        return imgs,label


    def get_labels(self, df):
        # Get the labels
        labels = df[RANZCR_TASKS]
        return labels
        
    def get_image(self, index):

        # Get and transform the label
        label = self.labels.iloc[index].values
        label = torch.FloatTensor(label)

        # Get and transform the image
        img_path = self.img_paths.iloc[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.return_info_dict:
            info_dict = {'paths': str(img_path)}
            return img, label, info_dict

        return img, label

    def __getitem__(self, index):
        # return self.get_image(index)
        if self.study_level:
            return self.get_validationImages(index)
        else:
            return self.get_image(index)
