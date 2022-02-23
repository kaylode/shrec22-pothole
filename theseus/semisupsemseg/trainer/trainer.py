import torch
import numpy as np
from torchvision.transforms import functional as TFF
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from .semisup_trainer import SemiSupervisedTrainer
from theseus.utilities.loading import load_state_dict
from theseus.utilities.visualization.visualizer import Visualizer
from theseus.utilities.visualization.colors import color_list
from theseus.utilities.analysis.analyzer import SegmentationAnalyzer
from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

class S4Trainer(SemiSupervisedTrainer):
    """Trainer for semi supervised semantic segmentation tasks
    
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def check_best(self, metric_dict):
        """
        Hook function, called after metrics are calculated
        """
        if metric_dict['dice'] > self.best_value:
            if self.iters > 0: # Have been training, else in evaluation-only mode or just sanity check
                LOGGER.text(
                    f"Evaluation improved from {self.best_value} to {metric_dict['dice']}",
                    level=LoggerObserver.INFO)
                self.best_value = metric_dict['dice']
                self.save_checkpoint('best')
            
            else:
                if self.visualize_when_val:
                    self.visualize_pred()

    def save_checkpoint(self, outname='last'):
        """
        Save all information of the current iteration
        """
        weights = {
            'model': self.model.model1.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'iters': self.iters,
            'best_value': self.best_value,
        }

        if self.scaler is not None:
            weights[self.scaler.state_dict_key] = self.scaler.state_dict()
           
        self.checkpoint.save(weights, outname)

    def load_checkpoint(self, path:str):
        """
        Load all information the current iteration from checkpoint 
        """
        LOGGER.text("Loading checkpoints...", level=LoggerObserver.INFO)
        state_dict = torch.load(path)
        self.epoch = load_state_dict(self.epoch, state_dict, 'epoch')
        self.start_iter = load_state_dict(self.start_iter, state_dict, 'iters')
        self.best_value = load_state_dict(self.best_value, state_dict, 'best_value')  
        self.scaler = load_state_dict(self.scaler, state_dict, self.scaler.state_dict_key)

        
    def visualize_gt(self):
        """
        Visualize dataloader for sanity check 
        """

        LOGGER.text("Visualizing dataset...", level=LoggerObserver.DEBUG)
        visualizer = Visualizer()
        batch = next(iter(self.trainloader))
        images = batch["inputs"]
        masks = batch['targets'].squeeze()

        batch = []
        for idx, (inputs, mask) in enumerate(zip(images, masks)):
            img_show = visualizer.denormalize(inputs)
            decode_mask = visualizer.decode_segmap(mask.numpy())
            img_show = TFF.to_tensor(img_show)
            decode_mask = TFF.to_tensor(decode_mask/255.0)
            img_show = torch.cat([img_show, decode_mask], dim=-1)
            batch.append(img_show)
        grid_img = visualizer.make_grid(batch)

        fig = plt.figure(figsize=(16,8))
        plt.axis('off')
        plt.imshow(grid_img)

        # segmentation color legends 
        classes = self.valloader.dataset.classnames
        patches = [mpatches.Patch(color=np.array(color_list[i][::-1]), 
                                label=classes[i]) for i in range(len(classes))]
        plt.legend(handles=patches, bbox_to_anchor=(-0.03, 1), loc="upper right", borderaxespad=0., 
                fontsize='large')
        plt.tight_layout(pad=0)

        LOGGER.log([{
            'tag': "Sanitycheck/batch/train",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': self.iters
            }
        }])

        
        batch = next(iter(self.valloader))
        images = batch["inputs"]
        masks = batch['targets'].squeeze()

        batch = []
        for idx, (inputs, mask) in enumerate(zip(images, masks)):
            img_show = visualizer.denormalize(inputs)
            decode_mask = visualizer.decode_segmap(mask.numpy())
            img_show = TFF.to_tensor(img_show)
            decode_mask = TFF.to_tensor(decode_mask/255.0)
            img_show = torch.cat([img_show, decode_mask], dim=-1)
            batch.append(img_show)
        grid_img = visualizer.make_grid(batch)

        fig = plt.figure(figsize=(16,8))
        plt.axis('off')
        plt.imshow(grid_img)
        plt.legend(handles=patches, bbox_to_anchor=(-0.03, 1), loc="upper right", borderaxespad=0., 
                fontsize='large')
        plt.tight_layout(pad=0)

        LOGGER.log([{
            'tag': "Sanitycheck/batch/val",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': self.iters
            }
        }])

    @torch.no_grad()
    def visualize_pred(self):
        r"""Visualize model prediction 
        
        """
        
        # Vizualize model predictions
        LOGGER.text("Visualizing model predictions...", level=LoggerObserver.DEBUG)

        visualizer = Visualizer()

        self.model.eval()

        batch = next(iter(self.valloader))
        images = batch["inputs"]
        masks = batch['targets'].squeeze()

        preds = self.model.model1.get_prediction(
            {'inputs': images, 'thresh': 0.5}, self.model.device)['masks']

        batch = []
        for idx, (inputs, mask, pred) in enumerate(zip(images, masks, preds)):
            img_show = visualizer.denormalize(inputs)
            decode_mask = visualizer.decode_segmap(mask.numpy())
            decode_pred = visualizer.decode_segmap(pred)
            img_cam = TFF.to_tensor(img_show)
            decode_mask = TFF.to_tensor(decode_mask/255.0)
            decode_pred = TFF.to_tensor(decode_pred/255.0)
            img_show = torch.cat([img_cam, decode_pred, decode_mask], dim=-1)
            batch.append(img_show)
        grid_img = visualizer.make_grid(batch)

        fig = plt.figure(figsize=(16,8))
        plt.axis('off')
        plt.title('Raw image - Prediction - Ground Truth')
        plt.imshow(grid_img)

        # segmentation color legends 
        classes = self.valloader.dataset.classnames
        patches = [mpatches.Patch(color=np.array(color_list[i][::-1]), 
                                label=classes[i]) for i in range(len(classes))]
        plt.legend(handles=patches, bbox_to_anchor=(-0.03, 1), loc="upper right", borderaxespad=0., 
                fontsize='large')
        plt.tight_layout(pad=0)

        LOGGER.log([{
            'tag': "Validation/prediction",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': self.iters
            }
        }])
        

    @torch.no_grad()
    def visualize_model(self):
        # Vizualize Model Graph
        LOGGER.text("Visualizing architecture...", level=LoggerObserver.DEBUG)

        batch = next(iter(self.valloader))
        images = batch["inputs"].to(self.model.device)
        LOGGER.log([{
            'tag': "Sanitycheck/analysis/architecture",
            'value': self.model.model,
            'type': LoggerObserver.TORCH_MODULE,
            'kwargs': {
                'inputs': images
            }
        }])

    def analyze_gt(self):
        """
        Perform simple data analysis
        """
        LOGGER.text("Analyzing datasets...", level=LoggerObserver.DEBUG)
        analyzer = SegmentationAnalyzer()
        analyzer.add_dataset(self.trainloader.dataset)
        fig = analyzer.analyze(figsize=(10,5))
        LOGGER.log([{
            'tag': "Sanitycheck/analysis/train",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': self.iters
            }
        }])

        analyzer = SegmentationAnalyzer()
        analyzer.add_dataset(self.valloader.dataset)
        fig = analyzer.analyze(figsize=(10,5))
        LOGGER.log([{
            'tag': "Sanitycheck/analysis/val",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': self.iters
            }
        }])

    def on_evaluate_end(self):
        if self.visualize_when_val:
            self.visualize_pred()
        self.save_checkpoint()
    
    def on_start(self):
        if self.resume is not None:
            self.load_checkpoint(self.resume)

    def sanitycheck(self):
        """Sanity check before training
        """
        self.visualize_gt()
        self.analyze_gt()
        self.visualize_model()
        self.evaluate_epoch()