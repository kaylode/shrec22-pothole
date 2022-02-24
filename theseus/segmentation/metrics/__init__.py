from theseus.base.metrics import METRIC_REGISTRY

from .dicecoeff import *
from .pixel_accuracy import *
from .mean_iou import *

METRIC_REGISTRY.register(PixelAccuracy)
METRIC_REGISTRY.register(DiceScore)
METRIC_REGISTRY.register(mIOU)