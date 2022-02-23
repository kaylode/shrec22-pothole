from theseus.base.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
from theseus.segmentation.datasets.mosaic_dataset import CSVDatasetWithMosaic
from .csv_dataset import CSVDataset

DATASET_REGISTRY.register(CSVDataset)
DATASET_REGISTRY.register(CSVDatasetWithMosaic)

from .cutmix_loader import CutmixLoader
DATALOADER_REGISTRY.register(CutmixLoader) 