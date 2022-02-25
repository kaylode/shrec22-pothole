from typing import Callable, Dict, Optional
from datetime import datetime

import os
import torch
from theseus.meanteacher.models.wrapper import TeacherStudentModel
from theseus.opt import Config
from theseus.base.optimizers import OPTIM_REGISTRY, SCHEDULER_REGISTRY
from theseus.meanteacher.augmentations import TRANSFORM_REGISTRY
from theseus.meanteacher.losses import LOSS_REGISTRY
from theseus.meanteacher.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
from theseus.meanteacher.trainer import TRAINER_REGISTRY
from theseus.meanteacher.metrics import METRIC_REGISTRY
from theseus.meanteacher.models import MODEL_REGISTRY
from theseus.utilities.getter import (get_instance, get_instance_recursively)
from theseus.utilities.loggers import LoggerObserver, TensorboardLogger, StdoutLogger, ImageWriter
from theseus.utilities.loading import load_state_dict, find_old_tflog

from theseus.utilities.cuda import get_devices_info



class Pipeline(object):
    """docstring for Pipeline."""

    def __init__(
        self,
        opt: Config
    ):
        super(Pipeline, self).__init__()
        self.opt = opt

        
        self.savedir = os.path.join(opt['global']['save_dir'], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.savedir, exist_ok=True)
        
        self.debug = opt['global']['debug']
        self.logger = LoggerObserver.getLogger("main") 

        stdout_logger = StdoutLogger(__name__, self.savedir, debug=self.debug)
        self.logger.subscribe(stdout_logger)
        self.logger.text(self.opt, level=LoggerObserver.INFO)

        self.use_fp16 = opt['global']['use_fp16']

        self.transform_cfg = Config.load_yaml(opt['global']['cfg_transform'])

        self.device_name = opt['global']['device']
        self.device = torch.device(self.device_name)
        self.resume = opt['global']['resume']
        self.pretrained = opt['global']['pretrained']

        self.transform = get_instance_recursively(
            self.transform_cfg, registry=TRANSFORM_REGISTRY
        )

        self.sup_train_dataset = get_instance_recursively(
            opt['data']["dataset"]['sup_train'],
            registry=DATASET_REGISTRY,
            transform=self.transform['train'],
        )

        self.unsup_train_dataset = get_instance_recursively(
            opt['data']["dataset"]['unsup_train'],
            registry=DATASET_REGISTRY,
            transform=self.transform['val'],
        )

        self.val_dataset = get_instance_recursively(
            opt['data']["dataset"]['val'],
            registry=DATASET_REGISTRY,
            transform=self.transform['val'],
        )

        CLASSNAMES = self.val_dataset.classnames

        self.sup_train_dataloader = get_instance(
            opt['data']["dataloader"]['sup_train'],
            registry=DATALOADER_REGISTRY,
            dataset=self.sup_train_dataset,
        )

        self.unsup_train_dataloader = get_instance(
            opt['data']["dataloader"]['unsup_train'],
            registry=DATALOADER_REGISTRY,
            dataset=self.unsup_train_dataset,
        )

        self.val_dataloader = get_instance(
            opt['data']["dataloader"]['val'],
            registry=DATALOADER_REGISTRY,
            dataset=self.val_dataset
        )

        model_s = get_instance(
          self.opt["model_s"], 
          registry=MODEL_REGISTRY, 
          classnames=CLASSNAMES,
          num_classes=len(CLASSNAMES)).to(self.device)

        model_t = get_instance(
          self.opt["model_t"], 
          registry=MODEL_REGISTRY, 
          classnames=CLASSNAMES,
          num_classes=len(CLASSNAMES)).to(self.device)
          
        sup_criterion = get_instance_recursively(
            self.opt["sup_loss"], 
            registry=LOSS_REGISTRY).to(self.device)
        
        csst_criterion = get_instance_recursively(
            self.opt["csst_loss"], 
            registry=LOSS_REGISTRY).to(self.device)

        if self.pretrained:
            state_dict = torch.load(self.pretrained)
            model_s = load_state_dict(model_s, state_dict, 'model')
        if self.resume:
            state_dict = torch.load(self.resume)
            model_s = load_state_dict(model_s, state_dict, 'model')

        self.model = TeacherStudentModel(
            model_s, 
            model_t,
            sup_criterion, 
            csst_criterion, 
            self.device)

        self.metrics = get_instance_recursively(
            self.opt['metrics'], 
            registry=METRIC_REGISTRY, 
            classnames=CLASSNAMES,
            num_classes=len(CLASSNAMES))

        self.optimizer = get_instance(
            self.opt["optimizer"],
            registry=OPTIM_REGISTRY,
            params=self.model.model_s.parameters(),
        )

        last_epoch = -1
        if self.resume:
            self.optimizer = load_state_dict(self.optimizer, state_dict, 'optimizer')
            last_epoch = load_state_dict(last_epoch, state_dict, 'epoch')

        self.scheduler = get_instance(
            self.opt["scheduler"], registry=SCHEDULER_REGISTRY, optimizer=self.optimizer,
            **{
                'num_epochs': self.opt["trainer"]['args']['num_epochs'],
                'trainset': self.sup_train_dataset,
                'batch_size': self.opt["data"]['dataloader']['val']['args']['batch_size'],
                'last_epoch': last_epoch,
            }
        )

        self.trainer = get_instance(
            self.opt["trainer"],
            model=self.model,
            suptrainloader=self.sup_train_dataloader,
            unsuptrainloader=self.unsup_train_dataloader,
            valloader=self.val_dataloader,
            metrics=self.metrics,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            use_fp16=self.use_fp16,
            save_dir=self.savedir,
            resume=self.resume,
            registry=TRAINER_REGISTRY,
        )

    def infocheck(self):
        self.logger.text(f"Number of trainable parameters: {self.model.trainable_parameters():,}", level=LoggerObserver.INFO)

        device_info = get_devices_info(self.device_name)
        self.logger.text("Using " + device_info, level=LoggerObserver.INFO)

        self.logger.text(f"Number of supervised training samples: {len(self.sup_train_dataset)}", level=LoggerObserver.INFO)
        self.logger.text(f"Number of unsupervised training samples: {len(self.unsup_train_dataset)}", level=LoggerObserver.INFO)
        self.logger.text(f"Number of validation samples: {len(self.val_dataset)}", level=LoggerObserver.INFO)
        self.logger.text(f"Number of supervised  training iterations each epoch: {len(self.sup_train_dataloader)}", level=LoggerObserver.INFO)
        self.logger.text(f"Number of unsupervised1 training iterations each epoch: {len(self.unsup_train_dataloader)}", level=LoggerObserver.INFO)
        self.logger.text(f"Number of unsupervised2 training iterations each epoch: {len(self.unsup_train_dataloader2)}", level=LoggerObserver.INFO)
        self.logger.text(f"Number of validation iterations each epoch: {len(self.val_dataloader)}", level=LoggerObserver.INFO)
        self.logger.text(f"Everything will be saved to {self.savedir}", level=LoggerObserver.INFO)

    def initiate(self):
        self.infocheck()

        self.opt.save_yaml(os.path.join(self.savedir, 'pipeline.yaml'))
        self.transform_cfg.save_yaml(os.path.join(self.savedir, 'transform.yaml'))

        tf_logger = TensorboardLogger(self.savedir)
        if self.resume is not None:
            tf_logger.load(find_old_tflog(
                os.path.dirname(os.path.dirname(self.resume))
            ))
        self.logger.subscribe(tf_logger)

        if self.debug:
            self.logger.text("Sanity checking before training...", level=LoggerObserver.DEBUG)
            self.trainer.sanitycheck()


    def fit(self):
        self.initiate()
        self.trainer.fit()

    def evaluate(self):
        self.infocheck()
        writer = ImageWriter(os.path.join(self.savedir, 'samples'))
        self.logger.subscribe(writer)

        self.logger.text("Evaluating...", level=LoggerObserver.INFO)
        self.trainer.evaluate_epoch()
   

  