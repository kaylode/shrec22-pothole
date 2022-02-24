from theseus.segmentation.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from .cutmix_loader import CutmixLoader
DATALOADER_REGISTRY.register(CutmixLoader) 