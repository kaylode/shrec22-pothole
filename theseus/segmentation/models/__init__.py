from theseus.base.models import MODEL_REGISTRY

from .segmodels import BaseSegModel
from .wrapper import ModelWithLoss
from .unet import EfficientUNet, R2UNet, ResidUNet
from .segformer import SegFormer

MODEL_REGISTRY.register(BaseSegModel)
MODEL_REGISTRY.register(SegFormer)
MODEL_REGISTRY.register(EfficientUNet)
MODEL_REGISTRY.register(R2UNet)
MODEL_REGISTRY.register(ResidUNet)
