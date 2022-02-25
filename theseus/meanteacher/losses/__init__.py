from theseus.segmentation.losses import LOSS_REGISTRY

from .mse_loss import MSELoss

LOSS_REGISTRY.register(MSELoss)