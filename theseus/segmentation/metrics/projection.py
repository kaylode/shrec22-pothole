import os
import torch
from typing import Any, Dict, Optional, List
from theseus.base.metrics.metric_template import Metric
import numpy as np
import hashlib
from theseus.utilities.visualization.visualizer import Visualizer
from theseus.utilities.loggers import LoggerObserver


class EmbeddingProjection(Metric):
    """
    Visualize embedding project for classification
    """
    def __init__(self, save_dir='.temp', **kwargs):
        super().__init__(**kwargs)
        self.save_dir = save_dir
        self.visualizer = Visualizer()
        self.logger = LoggerObserver.getLogger('main')
        self.reset()

        os.makedirs(self.save_dir, exist_ok=True)

    def update(self, outputs: torch.Tensor, batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        features = outputs.detach().cpu().numpy() # (B, C, H, W)
        img_names = batch['img_names']

        for i in range(len(features)):
            filename = hashlib.md5(img_names[i].encode('utf-8')).hexdigest()
            embedding_path = self.save_dir + r"/" + filename + '_feat.npy' 
            np.save(embedding_path, features[i])
            self.embeddings.append(embedding_path)
       
    def reset(self):
        self.embeddings = []

    def value(self):
        return {'projection': "Saved prediction as numy"}