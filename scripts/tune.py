import os
import parser
import re
import subprocess
from argparse import ArgumentParser

import joblib
import optuna

SUBMISSION_FOLDER = ""
NUM_MODELS = 1

import random


import torch
import hashlib
import numpy as np
from theseus.opt import Opts
from theseus.opt import Config
from theseus.cps.metrics import METRIC_REGISTRY
from theseus.cps.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
from theseus.utilities.getter import (get_instance, get_instance_recursively)
from theseus.utilities.loggers import LoggerObserver, TensorboardLogger, StdoutLogger, ImageWriter

class TuningPipeline(object):
    """docstring for TuningPipeline."""

    def __init__(
        self,
        opt: Config
    ):
        super(TuningPipeline, self).__init__()
        self.opt = opt

        self.numpy_dirs = opt['global']['numpy_dirs']
        self.savedir = opt['global']['save_dir']
        os.makedirs(self.savedir, exist_ok=True)
        
        self.debug = opt['global']['debug']
        self.logger = LoggerObserver.getLogger("main") 

        stdout_logger = StdoutLogger(__name__, self.savedir, debug=self.debug)
        self.logger.subscribe(stdout_logger)
        self.logger.text(self.opt, level=LoggerObserver.INFO)
        self.transform_cfg = Config.load_yaml(opt['global']['cfg_transform'])

        self.val_dataset = get_instance_recursively(
            opt['data']["dataset"]['val'],
            registry=DATASET_REGISTRY,
            transform=self.transform['val'],
        )

        CLASSNAMES = self.val_dataset.classnames

        self.metrics = get_instance_recursively(
            self.opt['metrics'], 
            registry=METRIC_REGISTRY, 
            classnames=CLASSNAMES,
            num_classes=len(CLASSNAMES))

        self.val_dataloader = get_instance(
            opt['data']["dataloader"]['val'],
            registry=DATALOADER_REGISTRY,
            dataset=self.val_dataset
        )

    def ensemble(self, logit, weights=None, reduction='max'):
        output = torch.softmax(logit, dim=1)# [num_models, C, H, W]

        if weights is not None:
            output = output * weights[:, None]

        if reduction == 'sum':
            output = output.sum(dim=0) #[C, H, W]
        elif reduction == 'max':
            output, _ = output.max(dim=0) #[C, H, W]

        ## Down't ask this neither
        # output[:,0] *= 0.25
        # output[:,1] *= 0.4
        # output[:,2] *= 0.35

        return output

    def evaluate(self, weights: np.ndarray):
        """
        Evaluate the model
        """

        for batch in self.val_dataloader:
            img_names = batch['img_names']

            batch_preds = []
            for i, item in enumerate(batch):
                filename = hashlib.md5(img_names[i].encode('utf-8')).hexdigest()

                all_embeddings = []
                for numpy_dir in self.numpy_dirs:
                    embedding_path = numpy_dir + r"/" + filename + '_feat.npy' 
                    embeddings = np.load(embedding_path)
                    all_embeddings.append(embeddings)

                ## Stack into tensors
                all_embeddings = torch.from_numpy(np.stack(all_embeddings, axis=0)) # (num_models, C, H, W)

                ensembled = self.ensemble(all_embeddings, reduction='sum', weights=weights)# (C, H, W)
                batch_preds.append(ensembled)

            batch_preds = torch.stack(batch_preds, dim=0) # (B, C, H, W)
        
            for metric in self.metrics:
                metric.update(batch_preds, batch)

        metric_dict = {}
        for metric in self.metrics:
            metric_dict.update(metric.value())
        return metric_dict


    def optim_function(self, params):
        """
        Optimize the model
        """
        # create a np vector from params
        w = np.array([params[f"w_{i}"] for i in range(NUM_MODELS)])
        score = self.evaluate(w)
        return score


    def objective(self, trial):
        params = {}
        for i in range(NUM_MODELS):
            params[f"w_{i}"] = trial.suggest_float(f"w_{i}", 0.01, 1)
        return self.optim_function(params)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--study-name", type=str, default="base")
    args = parser.parse_args()
    config = Config(args.yaml_path)

    NUM_MODELS = args.n
    study = optuna.create_study(direction="maximize", study_name=args.study_name)

    val_pipeline = TuningPipeline(config)
    study.optimize(val_pipeline.objective, n_trials=100)

    print(study.best_params)
    joblib.dump(study, f"{args.study_name}.pkl")

