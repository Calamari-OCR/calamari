import copy

import numpy as np
from calamari_ocr.ocr.augmentation import SimpleDataAugmenter
from tfaip.base.data.pipeline.dataprocessor import DataProcessor
from tfaip.base.data.pipeline.definitions import InputTargetSample


class AugmentationProcessor(DataProcessor):
    @staticmethod
    def default_params() -> dict:
        return {'augmenter_type': 'simple'}

    def __init__(self, augmenter_type, **kwargs):
        super(AugmentationProcessor, self).__init__(**kwargs)
        assert(augmenter_type == 'simple')
        self.data_augmenter = SimpleDataAugmenter()

    def apply(self, line, text, meta: dict):
        # data augmentation
        if not self.params.data_aug_params.no_augs() \
                and line is not None \
                and self.data_augmenter \
                and np.random.rand() <= self.params.data_aug_params.to_rel():
            line, text = self.augment(line, text, meta)
        return line, text

    def augment(self, line, text, meta):
        meta['augmented'] = True
        return self.data_augmenter.augment_single(line, text)

    def multi_augment(self, sample: InputTargetSample, n_augmentations=1, include_non_augmented=True):
        if include_non_augmented:
            out = [sample]
        else:
            out = []

        for n in range(n_augmentations):
            meta = copy.deepcopy(sample.meta)
            l, t = self.augment(sample.inputs, sample.targets, meta)
            out.append(InputTargetSample(l, t, meta))

        return out

