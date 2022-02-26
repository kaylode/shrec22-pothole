import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Dict, Any

class TeacherStudentModel(nn.Module):
    """Add utilitarian functions for module to work with pipeline

    Args:
        model (Module): Base Model without loss
        loss (Module): Base loss function with stat

    """

    def __init__(
        self, 
        model_s: nn.Module, 
        model_t: nn.Module, 
        criterion_sup: nn.Module, 
        criterion_csst: nn.Module, 
        device: torch.device,
        ema_decay: float = 0.99,
        weights: List[float] = [1.0, 100.0]):

        super().__init__()
        self.model_s = model_s
        self.model_t = model_t
        self.criterion_sup = criterion_sup
        self.criterion_csst = criterion_csst
        self.device = device
        self.ema_decay = ema_decay
        self.weights = weights
        self.num_classes = self.model_s.num_classes

        # detach the teacher model
        for param in self.model_t.parameters():
            param.detach_()

        for t_param, s_param in zip(self.model_t.parameters(), self.model_s.parameters()):
            t_param.data.copy_(s_param.data)

    def _update_ema_variables(self, ema_decay, global_step):
        ema_decay = min(1 - 1 / (global_step + 1), ema_decay)
        for t_param, s_param in zip(self.model_t.parameters(), self.model_s.parameters()):
            t_param.data.mul_(ema_decay).add_(s_param.data * (1 - ema_decay))

    def forward(self, batch, metrics=None):
        inputs = batch["inputs"].to(self.device)
        outputs = self.model_s(inputs)
            
        loss, loss_dict = self.criterion_sup(outputs, batch, self.device)

        if metrics is not None:
            for metric in metrics:
                metric.update(outputs, batch)

        return {
            'loss': loss,
            'loss_dict': loss_dict
        }

    def training_step(self, sup_batch, unsup_batch, global_Step):
        sup_inputs = sup_batch['inputs'].to(self.device)
        unsup_inputs = unsup_batch['inputs'].to(self.device)
        
        ## Get student predictions for inputs
        s_sup_probs = self.model_s(sup_inputs)
        s_unsup_probs = self.model_s(unsup_inputs)

        ## Get teacher predictions for inputs
        with torch.no_grad():
            t_sup_probs = self.model_t(sup_inputs)
            t_unsup_probs = self.model_t(unsup_inputs)

        ## EMA update teacher model
        self._update_ema_variables(self.ema_decay, global_Step=global_Step)

        ## Concatenate outputs
        s_probs = torch.cat([s_sup_probs, s_unsup_probs], dim=0)
        t_probs = torch.cat([t_sup_probs, t_unsup_probs], dim=0)

        ## Get softmax normalization
        softmax_pred_s = F.softmax(s_probs, dim=1)
        softmax_pred_t = F.softmax(t_probs, dim=1)

        ## Mean teacher loss
        csst_loss, _ = self.criterion_csst(softmax_pred_s, {'targets':softmax_pred_t.detach()}, self.device)
        csst_loss = self.weights[1] * csst_loss

        # Supervised loss

        ## Student loss
        s_sup_loss, _ = self.criterion_sup(s_sup_probs, sup_batch, self.device)
        s_sup_loss = self.weights[0] * s_sup_loss

        ## Teacher loss. No backward
        t_sup_loss, _ = self.criterion_sup(t_sup_probs, sup_batch, self.device)

        # Total loss
        loss = s_sup_loss + csst_loss
        loss_dict = {
            'ST': s_sup_loss.item(),
            'TC': t_sup_loss.item(),
            'CSST': csst_loss.item(),
            'Total': loss.item()  
        }

        return {
            'loss': loss,
            'loss_dict': loss_dict
        }

    def evaluate_step(self, batch, metrics=None):
        return self.forward(batch, metrics)

    def state_dict(self):
        return self.model_s.state_dict()

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model(self):
        return self

    @torch.no_grad()
    def get_prediction(self, adict: Dict[str, Any], device: torch.device):
        inputs = adict['inputs'].to(device)
        probs = self.model_s(inputs)
        predict = torch.argmax(probs, dim=1)
        predict = predict.detach().cpu().squeeze().numpy()
        return {
            'masks': predict
        } 
