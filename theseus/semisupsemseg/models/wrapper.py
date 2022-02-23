import torch
from torch import nn
from typing import List

class ModelWithLoss(nn.Module):
    """Add utilitarian functions for module to work with pipeline

    Args:
        model (Module): Base Model without loss
        loss (Module): Base loss function with stat

    """

    def __init__(
        self, 
        model1: nn.Module, 
        model2: nn.Module, 
        criterion_sup: nn.Module, 
        criterion_unsup: nn.Module, 
        weights: List,
        device: torch.device):

        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.criterion_sup = criterion_sup
        self.criterion_unsup = criterion_unsup
        self.device = device
        self.weights = weights

    def forward(self, batch, metrics=None):
        outputs = self.model1(batch["inputs"].to(self.device))
        loss, loss_dict = self.criterion_sup(outputs, batch, self.device)

        if metrics is not None:
            for metric in metrics:
                metric.update(outputs, batch)

        return {
            'loss': loss,
            'loss_dict': loss_dict
        }

    def training_step(self, sup_batch, unsup_batch1, unsup_batch2):
        sup_inputs = sup_batch['inputs'].to(self.device)
        unsup_inputs1 = unsup_batch1['inputs'].to(self.device)
        unsup_inputs2 = unsup_batch2['inputs'].to(self.device)

        #Unsupervised loss
        batch_mix_masks = unsup_inputs1['cutmix_masks']
        unsup_imgs_mixed = unsup_inputs1 * (1 - batch_mix_masks) + unsup_inputs2 * batch_mix_masks
        with torch.no_grad():
            ## Estimate the pseudo-label with branch#1 & supervise branch#2
            logits_u0_tea_1 = self.model1(unsup_inputs1)
            logits_u1_tea_1 = self.model1(unsup_inputs2)
            logits_u0_tea_1 = logits_u0_tea_1.detach()
            logits_u1_tea_1 = logits_u1_tea_1.detach()
            ## Estimate the pseudo-label with branch#2 & supervise branch#1
            logits_u0_tea_2 = self.model2(unsup_inputs1)
            logits_u1_tea_2 = self.model2(unsup_inputs2)
            logits_u0_tea_2 = logits_u0_tea_2.detach()
            logits_u1_tea_2 = logits_u1_tea_2.detach()

        logits_cons_tea_1 = logits_u0_tea_1 * (1 - batch_mix_masks) + logits_u1_tea_1 * batch_mix_masks
        _, ps_label_1 = torch.max(logits_cons_tea_1, dim=1)
        ps_label_1 = ps_label_1.long()
        logits_cons_tea_2 = logits_u0_tea_2 * (1 - batch_mix_masks) + logits_u1_tea_2 * batch_mix_masks
        _, ps_label_2 = torch.max(logits_cons_tea_2, dim=1)
        ps_label_2 = ps_label_2.long()

        ## Get student#1 prediction for mixed image
        logits_cons_stu_1 = self.model1(unsup_imgs_mixed)
        ## Get student#2 prediction for mixed image
        logits_cons_stu_2 = self.model2(unsup_imgs_mixed)

        cps_loss = self.criterion_unsup(logits_cons_stu_1, ps_label_2) + self.criterion_unsup(logits_cons_stu_2, ps_label_1)

        # Supervised loss
        sup_pred_1 = self.model1(sup_inputs)
        sup_pred_2 = self.model2(sup_inputs)

        sup_loss1, _ = self.criterion_sup(sup_pred_1, sup_batch)
        sup_loss2, _ = self.criterion_sup(sup_pred_2, sup_batch)
        sup_loss = sup_loss1 + sup_loss2

        # Total loss
        loss = self.weights[0] * sup_loss + self.weights[1] * cps_loss
        loss_dict = {
            'SUP1': sup_loss1.item(),
            'SUP2': sup_loss2.item(),
            'CPS': cps_loss.item(),
            'Total': loss.item()  
        }

        return {
            'loss': loss,
            'loss_dict': loss_dict
        }

    def evaluate_step(self, batch, metrics=None):
        return self.forward(batch, metrics)

    def state_dict(self):
        return self.model1.state_dict()

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)