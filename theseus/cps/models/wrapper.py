import torch
from torch import nn
from typing import List, Dict, Any
import torch.nn.functional as F

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
        device: torch.device,
        soft_cps: bool = False,
        weights: List[float] = [0.8, 0.2]):

        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.criterion_sup = criterion_sup
        self.criterion_unsup = criterion_unsup
        self.device = device
        self.weights = weights
        self.soft_cps = soft_cps
        self.num_classes = self.model1.num_classes

    def forward(self, batch, metrics=None):
        inputs = batch["inputs"].to(self.device)
        outputs1 = self.model1(inputs)
        outputs2 = self.model2(inputs)
        probs = self.ensemble_learning(outputs1, outputs2)
            
        loss, loss_dict = self.criterion_sup(probs, batch, self.device)

        if metrics is not None:
            for metric in metrics:
                metric.update(probs, batch)

        return {
            'loss': loss,
            'loss_dict': loss_dict
        }

    def training_step(self, sup_batch, unsup_batch1, unsup_batch2):
        sup_inputs = sup_batch['inputs'].to(self.device)
        unsup_inputs1 = unsup_batch1['inputs'].to(self.device)
        unsup_inputs2 = unsup_batch2['inputs'].to(self.device)

        #Unsupervised loss
        # batch_mix_masks = unsup_batch1['cutmix_masks'].to(self.device)
        # unsup_imgs_mixed = unsup_inputs1 * (1 - batch_mix_masks) + unsup_inputs2 * batch_mix_masks
        with torch.no_grad():
            ## Estimate the pseudo-label with branch#1 & supervise branch#2
            logits_u0_tea_1 = self.model1(unsup_inputs1)
            # logits_u1_tea_1 = self.model1(unsup_inputs2)
            logits_u0_tea_1 = logits_u0_tea_1.detach()
            # logits_u1_tea_1 = logits_u1_tea_1.detach()
            ## Estimate the pseudo-label with branch#2 & supervise branch#1
            # logits_u0_tea_2 = self.model2(unsup_inputs1)
            logits_u1_tea_2 = self.model2(unsup_inputs2)
            # logits_u0_tea_2 = logits_u0_tea_2.detach()
            logits_u1_tea_2 = logits_u1_tea_2.detach()

        if self.soft_cps:
            logits_cons_tea_1 = logits_u0_tea_1 #* (1 - batch_mix_masks) + logits_u1_tea_1 * batch_mix_masks
            one_hot_ps_label_1 = F.softmax(logits_cons_tea_1, dim=1)

            logits_cons_tea_2 = logits_u1_tea_2 #* (1 - batch_mix_masks) + logits_u1_tea_2 * batch_mix_masks
            one_hot_ps_label_2 = F.softmax(logits_cons_tea_2, dim=1)

            ## Don't ask what these lines mean
            one_hot_ps_label_2[:,0] *= 0.25
            one_hot_ps_label_2[:,1] *= 0.4
            one_hot_ps_label_2[:,2] *= 0.35

            one_hot_ps_label_1[:,0] *= 0.3
            one_hot_ps_label_1[:,1] *= 0.35
            one_hot_ps_label_1[:,2] *= 0.35

        else:
            logits_cons_tea_1 = logits_u0_tea_1 #* (1 - batch_mix_masks) + logits_u1_tea_1 * batch_mix_masks
            _, ps_label_1 = torch.max(logits_cons_tea_1, dim=1)
            ps_label_1 = ps_label_1.long()
            logits_cons_tea_2 = logits_u1_tea_2 #* (1 - batch_mix_masks) + logits_u1_tea_2 * batch_mix_masks
            _, ps_label_2 = torch.max(logits_cons_tea_2, dim=1)
            ps_label_2 = ps_label_2.long()

            ## One-hot encoding
            one_hot_ps_label_1 = torch.nn.functional.one_hot(
                ps_label_1, 
                num_classes=self.num_classes).permute(0, 3, 1, 2)
            one_hot_ps_label_2 = torch.nn.functional.one_hot(
                ps_label_2, 
                num_classes=self.num_classes).permute(0, 3, 1, 2)

        ## Get student#1 prediction for mixed image
        logits_cons_stu_1 = self.model1(unsup_inputs2)
        ## Get student#2 prediction for mixed image
        logits_cons_stu_2 = self.model2(unsup_inputs1)

        cps_loss1, _ = self.criterion_unsup(
            logits_cons_stu_1, {'targets': one_hot_ps_label_2.float()}, self.device)
        cps_loss2, _ = self.criterion_unsup(
            logits_cons_stu_2, {'targets': one_hot_ps_label_1.float()}, self.device)
        cps_loss1 = self.weights[1] * cps_loss1
        cps_loss2 = self.weights[1] * cps_loss2
        cps_loss = cps_loss1 + cps_loss2

        # Supervised loss
        sup_pred_1 = self.model1(sup_inputs)
        sup_pred_2 = self.model2(sup_inputs)

        sup_loss1, _ = self.criterion_sup(sup_pred_1, sup_batch, self.device)
        sup_loss2, _ = self.criterion_sup(sup_pred_2, sup_batch, self.device)
        sup_loss1 = self.weights[0] * sup_loss1
        sup_loss2 = self.weights[0] * sup_loss2
        sup_loss = sup_loss1 + sup_loss2

        # Total loss
        loss = sup_loss + cps_loss
        loss_dict = {
            'SUP1': sup_loss1.item(),
            'SUP2': sup_loss2.item(),
            'CPS1': cps_loss1.item(),
            'CPS2': cps_loss2.item(),
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

    def get_model(self):
        return self

    def ensemble_learning(self, logit1, logit2, reduction='max'):
        prob1 = torch.softmax(logit1, dim=1)
        prob2 = torch.softmax(logit2, dim=1)

        output = torch.stack([prob1, prob2], dim=0) # [2, B, C, H, W]
        if reduction == 'sum':
            output = output.sum(dim=0) #[B, C, H, W]
        elif reduction == 'max':
            output, _ = output.max(dim=0) #[B, C, H, W]

        ## Down't ask this neither
        output[:,0] *= 0.25
        output[:,1] *= 0.4
        output[:,2] *= 0.35

        return output

    @torch.no_grad()
    def get_prediction(self, adict: Dict[str, Any], device: torch.device):
        inputs = adict['inputs'].to(device)
        outputs1 = self.model1(inputs)
        outputs2 = self.model2(inputs)

        probs = self.ensemble_learning(outputs1, outputs2)
        predict = torch.argmax(probs, dim=1)

        predict = predict.detach().cpu().squeeze().numpy()
        return {
            'masks': predict
        } 
