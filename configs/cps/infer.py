from typing import List, Optional, Tuple

import matplotlib as mpl
mpl.use("Agg")
from theseus.opt import Opts

import os
import cv2
import torch
import numpy as np
from datetime import datetime
from theseus.opt import Config
from theseus.cps.models import MODEL_REGISTRY
from theseus.cps.augmentations import TRANSFORM_REGISTRY
from theseus.cps.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from theseus.utilities.loading import load_state_dict
from theseus.utilities.loggers import LoggerObserver, StdoutLogger
from theseus.utilities.cuda import get_devices_info
from theseus.utilities.getter import (get_instance, get_instance_recursively)
from theseus.utilities.visualization.visualizer import Visualizer
from theseus.cps.models.wrapper import ModelWithLoss

class TestPipeline(object):
    def __init__(
            self,
            opt: Config
        ):

        super(TestPipeline, self).__init__()
        self.opt = opt

        self.debug = opt['global']['debug']
        self.logger = LoggerObserver.getLogger("main") 
        self.savedir = os.path.join(opt['global']['save_dir'], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.savedir, exist_ok=True)

        stdout_logger = StdoutLogger(__name__, self.savedir, debug=self.debug)
        self.logger.subscribe(stdout_logger)
        self.logger.text(self.opt, level=LoggerObserver.INFO)

        self.transform_cfg = Config.load_yaml(opt['global']['cfg_transform'])
        self.device_name = opt['global']['device']
        self.device = torch.device(self.device_name)

        self.weights = opt['global']['weights']

        self.transform = get_instance_recursively(
            self.transform_cfg, registry=TRANSFORM_REGISTRY
        )

        self.dataset = get_instance(
            opt['data']["dataset"],
            registry=DATASET_REGISTRY,
            transform=self.transform['val'],
        )
        CLASSNAMES = self.dataset.classnames

        self.dataloader = get_instance(
            opt['data']["dataloader"],
            registry=DATALOADER_REGISTRY,
            dataset=self.dataset,
        )

        self.model1 = get_instance(
          self.opt["model1"], 
          registry=MODEL_REGISTRY, 
          classnames=CLASSNAMES,
          num_classes=len(CLASSNAMES)).to(self.device)

        self.model2 = get_instance(
          self.opt["model2"], 
          registry=MODEL_REGISTRY, 
          classnames=CLASSNAMES,
          num_classes=len(CLASSNAMES)).to(self.device)

        if self.weights:
            state_dict = torch.load(self.weights)
            self.model1 = load_state_dict(self.model1, state_dict, 'model1')
            self.model2 = load_state_dict(self.model2, state_dict, 'model2')

        self.model = ModelWithLoss(
            self.model1, 
            self.model2,
            criterion_sup=None, 
            criterion_unsup=None, 
            device=self.device)

    
    def infocheck(self):
        device_info = get_devices_info(self.device_name)
        self.logger.text("Using " + device_info, level=LoggerObserver.INFO)
        self.logger.text(f"Number of test sample: {len(self.dataset)}", level=LoggerObserver.INFO)
        self.logger.text(f"Everything will be saved to {self.savedir}", level=LoggerObserver.INFO)

    @torch.no_grad()
    def inference(self):
        self.infocheck()
        self.logger.text("Inferencing...", level=LoggerObserver.INFO)

        visualizer = Visualizer()
        self.model.eval()

        saved_mask_dir = os.path.join(self.savedir, 'masks')
        saved_overlay_dir = os.path.join(self.savedir, 'overlays')

        os.makedirs(saved_mask_dir, exist_ok=True)
        os.makedirs(saved_overlay_dir, exist_ok=True)

        for idx, batch in enumerate(self.dataloader):
            inputs = batch['inputs']
            img_names = batch['img_names']
            ori_sizes = batch['ori_sizes']

            outputs = self.model.get_prediction(batch, self.device)
            preds = outputs['masks']

            for (input, pred, filename, ori_size) in zip(inputs, preds, img_names, ori_sizes):
                decode_pred = visualizer.decode_segmap(pred)[:,:,::-1]
                resized_decode_mask = cv2.resize(decode_pred, dsize=tuple(ori_size))

                # Save mask
                savepath = os.path.join(saved_mask_dir, filename)
                cv2.imwrite(savepath, resized_decode_mask)

                # Save overlay
                raw_image = visualizer.denormalize(input)   
                raw_image = (raw_image*255).astype(np.uint8)
                ori_image = cv2.resize(raw_image, dsize=tuple(ori_size))
                ori_image = cv2.cvtColor(ori_image, cv2.COLOR_RGB2BGR)
                overlay = ori_image * 0.75 + resized_decode_mask * 0.25
                savepath = os.path.join(saved_overlay_dir, filename)
                cv2.imwrite(savepath, overlay)

                self.logger.text(f"Save image at {savepath}", level=LoggerObserver.INFO)
        

if __name__ == '__main__':
    opts = Opts().parse_args()
    val_pipeline = TestPipeline(opts)
    val_pipeline.inference()

        
