from theseus.segmentation.trainer import TRAINER_REGISTRY 

from .trainer import S4Trainer

TRAINER_REGISTRY.register(S4Trainer)