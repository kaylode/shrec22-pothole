import torch
from torch import nn
from typing import List, Dict, Any
import torch.nn.functional as F

class SemanticEnsembler(nn.Module):
    """Add utilitarian functions for module to work with pipeline

    Args:
        model (Module): Base Model without loss
        loss (Module): Base loss function with stat

    """

    def __init__(
        self, 
        models: List[torch.nn.Module],
        weights: List[float] = None,
        reduction='sum',
        **kwargs):

        models = nn.ModuleList(models)
        super().__init__()
        self.models = models
        self.weights = weights
        self.reduction = reduction

    def ensemble_inference(self, logits, reduction='max', weights = None):
        
        assert len(weights) == len(logits), "Number of weights must match number of models"
        probs = []
        for i, logit in enumerate(logits):
            prob = torch.softmax(logit, dim=1)
            if weights is not None:
                prob *= weights[i]
            probs.append(prob)

        output = torch.stack(probs, dim=0) # [N, B, C, H, W]

        if reduction == 'sum':
            output = output.sum(dim=0) #[B, C, H, W]
        elif reduction == 'max':
            output, _ = output.max(dim=0) #[B, C, H, W]

        ## Don't ask this neither
        output[:,0] *= 0.25
        output[:,1] *= 0.4
        output[:,2] *= 0.35

        return output

    @torch.no_grad()
    def get_prediction(self, adict: Dict[str, Any], device: torch.device):
        inputs = adict['inputs'].to(device)

        outputs_list = []
        for model in self.models:
            outputs = model(inputs)
            outputs_list.append(outputs)

        probs = self.ensemble_inference(outputs_list, self.reduction, self.weights)
        predict = torch.argmax(probs, dim=1)

        predict = predict.detach().cpu().squeeze().numpy()
        return {
            'masks': predict
        } 
