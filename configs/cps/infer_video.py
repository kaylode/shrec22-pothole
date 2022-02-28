from typing import List, Optional, Tuple

import matplotlib as mpl
mpl.use("Agg")
from theseus.opt import Opts

import os
import cv2
import torch
import numpy as np
from theseus.opt import Config
from theseus.segmentation.models import MODEL_REGISTRY
from theseus.segmentation.augmentations import TRANSFORM_REGISTRY
from theseus.segmentation.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from theseus.utilities.loading import load_state_dict
from theseus.utilities.loggers import LoggerObserver, StdoutLogger
from theseus.utilities.cuda import get_devices_info
from theseus.utilities.getter import (get_instance, get_instance_recursively)

from theseus.utilities.visualization.visualizer import Visualizer
from theseus.cps.models.wrapper import ModelWithLoss

class VideoWriter:
    def __init__(self, video_info, saved_path):
        self.video_info = video_info
        self.saved_path = saved_path

        if not os.path.exists(self.saved_path):
            os.makedirs(self.saved_path, exist_ok=True)
            
        video_name = self.video_info['name']
        outpath =os.path.join(self.saved_path, video_name)
        self.FPS = self.video_info['fps']
        self.WIDTH = self.video_info['width']
        self.HEIGHT = self.video_info['height']
        self.NUM_FRAMES = self.video_info['num_frames']
        self.outvid = cv2.VideoWriter(
            outpath,   
            cv2.VideoWriter_fourcc(*'mp4v'), 
            self.FPS, 
            (self.WIDTH, self.HEIGHT))

    def write(self, frame):
        self.outvid.write(frame)

class TestPipeline(object):
    def __init__(
            self,
            opt: Config
        ):

        super(TestPipeline, self).__init__()
        self.opt = opt

        self.debug = opt['global']['debug']
        self.logger = LoggerObserver.getLogger("main") 
        self.savedir = opt['global']['save_dir']
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
            soft_cps=True,
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

        video_name = os.path.basename(self.dataset.video_path)
        saved_mask_path = os.path.join(self.savedir, f'{video_name}_masks.mp4')
        saved_overlay_path = os.path.join(self.savedir, f'{video_name}_overlay.mp4')

        mask_writer = VideoWriter(self.dataset.video_info, saved_mask_path)
        overlay_writer = VideoWriter(self.dataset.video_info, saved_overlay_path)

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
                mask_writer.write(resized_decode_mask)

                # Save overlay
                raw_image = visualizer.denormalize(input)   
                raw_image = (raw_image*255).astype(np.uint8)
                ori_image = cv2.resize(raw_image, dsize=tuple(ori_size))
                ori_image = cv2.cvtColor(ori_image, cv2.COLOR_RGB2BGR)
                overlay = ori_image * 0.75 + resized_decode_mask * 0.25
                overlay_writer.write(overlay)

        self.logger.text(f"Save submission video at {saved_mask_path}", level=LoggerObserver.INFO)
        self.logger.text(f"Save overlay video at {saved_overlay_path}", level=LoggerObserver.INFO)
        

if __name__ == '__main__':
    opts = Opts().parse_args()
    val_pipeline = TestPipeline(opts)
    val_pipeline.inference()

        
