import numpy as np
import torch

from .mask_gen import BoxMaskGenerator

class CutmixCollateWrapper(object):
    def __init__(self, batch_aug_fn=None):
        self.batch_aug_fn = batch_aug_fn
        self.mask_generator = BoxMaskGenerator(
            prop_range = (0.25, 0.5),
            n_boxes = 3,
            random_aspect_ratio = True,
            prop_by_area = False,
            within_bounds = False,
            invert = True
        )

    def _generate_cutmix_mask(self, num_masks: int, width: int, height: int):
        return self.mask_generator.generate_params(num_masks, (width, height))

    def __call__(self, batch):
        if self.batch_aug_fn is not None:
            batch = self.batch_aug_fn(batch)
        batch_size, _, w, h = batch['inputs'].shape
        masks = self._generate_cutmix_mask(batch_size, w, h)
        batch['cutmix_masks'] = torch.from_numpy(masks.astype(np.float32))
        return batch

class CutmixLoader(torch.utils.data.DataLoader):
    r"""CutmixLoader, also return cutmix mask
    
    dataset: `torch.utils.data.Dataset`
        dataset, must have classes_dict and collate_fn attributes
    batch_size: `int`
        number of samples in one batch
    """
    def __init__(self, 
        dataset: torch.utils.data.Dataset, 
        batch_size: int, 
        **kwargs):

        if hasattr(dataset, 'collate_fn'):
            pre_collate_fn = dataset.collate_fn
        else:
            pre_collate_fn = None

        collate_fn = CutmixCollateWrapper(pre_collate_fn)        
        
        super(CutmixLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            collate_fn = collate_fn,
            **kwargs
        )