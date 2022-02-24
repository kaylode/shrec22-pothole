from typing import List, Optional, Tuple
import torch
from torch.cuda import amp

import os
import time
import numpy as np
from tqdm import tqdm
from theseus.utilities.loggers.cp_logger import Checkpoint
from theseus.base.optimizers.scalers import NativeScaler

from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

class SemiSupervisedTrainer(object):
    """Trainer for SemiSupervised tasks
    
    """
    def __init__(self,
                model, 
                suptrainloader,
                unsuptrainloader1, 
                unsuptrainloader2, 
                valloader,
                metrics,
                optimizer,
                scheduler,
                save_dir: str = 'runs',
                use_fp16: bool = False, 
                num_epochs: int = 100,
                num_iter_per_epoch: int = 1000,
                total_accumulate_steps: Optional[int] = None,
                clip_grad: float = 10.0,
                print_per_iter: int = 100,
                save_per_iter: int = 100,
                evaluate_per_epoch: int = 1,
                visualize_when_val: bool = True,
                best_value: float = 0.0,
                resume: str = Optional[None],
                ):
        
        self.model = model
        self.metrics = metrics 
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.suptrainloader = suptrainloader
        self.unsuptrainloader1 = unsuptrainloader1
        self.unsuptrainloader2 = unsuptrainloader2
        self.valloader = valloader

        self.save_dir = save_dir
        self.checkpoint = Checkpoint(os.path.join(self.save_dir, 'checkpoints'))
        self.num_epochs = num_epochs
        self.num_iter_per_epoch = num_iter_per_epoch
        self.step_per_epoch = self.scheduler.step_per_epoch
        self.use_amp = True if use_fp16 else False
        self.scaler = NativeScaler() if use_fp16 else False

        if total_accumulate_steps is None:
            self.accumulate_steps = 1
        else:
            self.accumulate_steps = max(round(total_accumulate_steps / suptrainloader.batch_size), 1) 
        self.clip_grad = clip_grad
        self.evaluate_per_epoch = evaluate_per_epoch
        self.print_per_iter = print_per_iter
        self.save_per_iter = save_per_iter
        self.visualize_when_val = visualize_when_val
        self.best_value = best_value
        self.resume = resume
        self.epoch = 0
        self.iters = 0
        self.start_iter = 0

    def fit(self): 
        # Total number of training iterations
        self.num_iters = (self.num_epochs+1) * self.num_iter_per_epoch
        
        # On start callbacks
        self.on_start()

        # Init scheduler params
        if self.step_per_epoch:
            self.scheduler.last_epoch = self.epoch - 1

        LOGGER.text(f'===========================START TRAINING=================================', level=LoggerObserver.INFO)
        for epoch in range(self.epoch, self.num_epochs):
            try:
                # Save current epoch
                self.epoch = epoch

                # Start training
                self.training_epoch()
                self.on_training_end()

                # Start evaluation
                if self.evaluate_per_epoch != 0:
                    if epoch % self.evaluate_per_epoch == 0 and epoch+1 >= self.evaluate_per_epoch:
                        self.evaluate_epoch(model_id=1)
                        self.evaluate_epoch(model_id=2)
                    self.on_evaluate_end()
                
                # On epoch end callbacks
                self.on_epoch_end()

            except KeyboardInterrupt:   
                break
        
        # On training finish callbacks
        self.on_finish()
        LOGGER.text("Training Completed!", level=LoggerObserver.INFO)

    def sanity_check(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError
        
    def visualize_batch(self):
        raise NotImplementedError

    def on_start(self):
        return

    def on_training_end(self):
        return

    def on_evaluate_end(self):
        return

    def on_epoch_end(self):
        if self.step_per_epoch:
            self.scheduler.step()
            lrl = [x['lr'] for x in self.optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            LOGGER.log([{
                'tag': 'Training/Learning rate',
                'value': lr,
                'type': LoggerObserver.SCALAR,
                'kwargs': {
                    'step': self.epoch
                }
            }])

    def on_finish(self):
        self.save_checkpoint()

    def training_epoch(self):
        """
        Perform training one epoch
        """
        self.model.train()

        suptrainloader = iter(self.suptrainloader)
        unsuptrainloader1 = iter(self.unsuptrainloader1)
        unsuptrainloader2 = iter(self.unsuptrainloader2)

        running_loss = {}
        running_time = 0

        self.optimizer.zero_grad()
        for i in range(self.num_iter_per_epoch):

            sup_batch = suptrainloader.next()
            unsup_batch1 = unsuptrainloader1.next()
            unsup_batch2 = unsuptrainloader2.next()
            
            start_time = time.time()

            loss = 0
            # Gradient scaler
            with amp.autocast(enabled=self.use_amp):
                outputs = self.model.training_step(sup_batch, unsup_batch1, unsup_batch2)
                loss = outputs['loss']
                loss_dict = outputs['loss_dict']
                loss /= self.accumulate_steps
                
            # Backward loss
            self.scaler(loss, self.optimizer)
            
            if i % self.accumulate_steps == 0 or i == len(self.trainloader)-1:
                self.scaler.step(self.optimizer, clip_grad=self.clip_grad, parameters=self.model.parameters())

                if not self.step_per_epoch:
                    self.scheduler.step()
                    lrl = [x['lr'] for x in self.optimizer.param_groups]
                    lr = sum(lrl) / len(lrl)

                    LOGGER.log([{
                        'tag': 'Training/Learning rate',
                        'value': lr,
                        'type': LoggerObserver.SCALAR,
                        'kwargs': {
                            'step': self.iters
                        }
                    }])


                self.optimizer.zero_grad()

            torch.cuda.synchronize()
            end_time = time.time()

            for (key,value) in loss_dict.items():
                if key in running_loss.keys():
                    running_loss[key] += value
                else:
                    running_loss[key] = value

            running_time += end_time-start_time

            # Calculate current iteration
            self.iters = self.start_iter + self.num_iter_per_epoch*self.epoch + i + 1

            # Logging
            if self.iters % self.print_per_iter == 0:
                for key in running_loss.keys():
                    running_loss[key] /= self.print_per_iter
                    running_loss[key] = np.round(running_loss[key], 5)
                loss_string = '{}'.format(running_loss)[1:-1].replace("'",'').replace(",",' ||')

                LOGGER.text(
                    "[{}|{}] [{}|{}] || {} || Time: {:10.4f}s".format(
                        self.epoch, self.num_epochs, self.iters, 
                        self.num_iters,loss_string, running_time), 
                    LoggerObserver.INFO)
                
                log_dict = [{
                    'tag': f"Training/{k} Loss",
                    'value': v/self.print_per_iter,
                    'type': LoggerObserver.SCALAR,
                    'kwargs': {
                        'step': self.iters
                    }
                } for k,v in running_loss.items()]
                LOGGER.log(log_dict)

                running_loss = {}
                running_time = 0

            # Saving checkpoint
            if (self.iters % self.save_per_iter == 0 or self.iters == self.num_iters - 1):
                LOGGER.text(f'Save model at [{self.iters}|{self.num_iters}] to last.pth', LoggerObserver.INFO)
                self.save_checkpoint()

    @torch.no_grad()   
    def evaluate_epoch(self, model_id=1):
        """
        Perform validation one epoch
        """
        self.model.eval()
        epoch_loss = {}

        metric_dict = {}
        LOGGER.text('=============================EVALUATION===================================', LoggerObserver.INFO)

        start_time = time.time()

        # Gradient scaler
        with amp.autocast(enabled=self.use_amp):
            for batch in tqdm(self.valloader):
                outputs = self.model.evaluate_step(batch, self.metrics, model_id=model_id)
                
                loss_dict = outputs['loss_dict']
                for (key,value) in loss_dict.items():
                    if key in epoch_loss.keys():
                        epoch_loss[key] += value
                    else:
                        epoch_loss[key] = value

        end_time = time.time()
        running_time = end_time - start_time
             
        metric_dict = {}
        for metric in self.metrics:
            metric_dict.update(metric.value())
            metric.reset()  

        # Logging
        for key in epoch_loss.keys():
            epoch_loss[key] /= len(self.valloader)
            epoch_loss[key] = np.round(epoch_loss[key], 5)
        loss_string = '{}'.format(epoch_loss)[1:-1].replace("'",'').replace(",",' ||')
        LOGGER.text(
            "[{}|{}] || {} || Time: {:10.4f} s".format(
                self.epoch, self.num_epochs, loss_string, running_time),
        level=LoggerObserver.INFO)

        metric_string = ""
        for metric, score in metric_dict.items():
            if isinstance(score, (int, float)):
                metric_string += metric +': ' + f"{score:.5f}" +' | '
        metric_string +='\n'

        LOGGER.text(metric_string, level=LoggerObserver.INFO)
        LOGGER.text('==========================================================================', level=LoggerObserver.INFO)

        log_dict = [{
            'tag': f"Validation/{k} Loss",
            'value': v/len(self.valloader),
            'type': LoggerObserver.SCALAR,
            'kwargs': {
                'step': self.epoch
            }
        } for k,v in epoch_loss.items()]

        log_dict += [{
            'tag': f"Validation/{k}",
            'value': v,
            'kwargs': {
                'step': self.epoch
            }
        } for k,v in metric_dict.items()]

        LOGGER.log(log_dict)

        # Hook function
        self.check_best(metric_dict)

    def check_best(self, metric_dict, model_id):
        return 