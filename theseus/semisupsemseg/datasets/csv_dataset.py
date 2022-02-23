from typing import Dict, List, Optional
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from .dataset import SegmentationDataset
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger('main')


class CSVDataset(SegmentationDataset):
    r"""CSVDataset multi-labels segmentation dataset

    Reads in .csv file with structure below:
        filename   | label
        ---------- | -----------
        <img1>.jpg | <mask1>.jpg

    image_dir: `str`
        path to directory contains images
    mask_dir: `str`
        path to directory contains masks
    transform: Optional[List]
        transformatin functions
        
    """
    def __init__(
            self, 
            image_dir: str, 
            mask_dir: str, 
            csv_path: str, 
            txt_classnames: str,
            transform: Optional[List] = None,
            **kwargs):
        super(CSVDataset, self).__init__(**kwargs)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.csv_path = csv_path
        self.transform = transform
        self.txt_classnames = txt_classnames
        self._load_data()

    def _load_data(self):
        """
        Read data from csv and load into memory
        """

        with open(self.txt_classnames, 'r') as f:
            self.classnames = f.read().splitlines()
        
        # Mapping between classnames and indices
        for idx, classname in enumerate(self.classnames):
            self.classes_idx[classname] = idx
        self.num_classes = len(self.classnames)
        
        df = pd.read_csv(self.csv_path)
        for idx, row in df.iterrows():
            img_name, mask_name = row
            image_path = os.path.join(self.image_dir,img_name)
            mask_path = os.path.join(self.mask_dir, mask_name)
            self.fns.append([image_path, mask_path])

    def _load_mask(self, label_path):
        mask = Image.open(label_path).convert('RGB')
        mask = np.array(mask)[:,:,::-1] # (H,W,3)
        mask = np.argmax(mask, axis=-1)  # (H,W) with each pixel value represent one class

        return mask 

    def _encode_masks(self, masks):
        """
        Input masks from _load_mask(), but in shape [B, H, W]
        Output should be one-hot encoding of segmentation masks [B, NC, H, W]
        """
        
        one_hot = torch.nn.functional.one_hot(masks.long(), num_classes=self.num_classes) # (B,H,W,NC)
        one_hot = one_hot.permute(0, 3, 1, 2) # (B,NC,H,W)
        return one_hot.float()
