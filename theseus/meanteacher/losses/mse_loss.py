from typing import Dict, List
from torch import nn

class MSELoss(nn.Module):
    r"""MSELoss is warper of mean squared error loss"""

    def __init__(self, reduction="mean", **kwargs):
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.criterion = nn.MSELoss(reduction=reduction)

    def forward(self, pred, batch, device):
        target = batch["targets"].to(device)
        loss = self.criterion(pred, target)
            
        loss_dict = {"MSE": loss.item()}
        return loss, loss_dict