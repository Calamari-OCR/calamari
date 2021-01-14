import copy
from functools import partial
from typing import List, Iterable

import numpy as np
from tfaip.util.multiprocessing.parallelmap import parallel_map

from calamari_ocr.ocr.augmentation import SimpleDataAugmenter
from tfaip.base.data.pipeline.dataprocessor import DataProcessor
from tfaip.base.data.pipeline.definitions import Sample


class AugmentationProcessor(DataProcessor):
    @staticmethod
    def default_params() -> dict:
        return {'augmenter_type': 'simple'}

    def __init__(self, augmenter_type, **kwargs):
        super(AugmentationProcessor, self).__init__(**kwargs)
        assert(augmenter_type == 'simple')
        self.data_augmenter = SimpleDataAugmenter()

    def preload(self,
                samples: List[Sample],
                num_processes=1,
                drop_invalid=True,
                progress_bar=False,
                ) -> Iterable[Sample]:
        n_augmentation = self.params.data_aug_params.to_abs()  # real number of augmentations
        if n_augmentation == 0:
            return samples

        apply_fn = partial(self.multi_augment, n_augmentations=n_augmentation, include_non_augmented=True)
        augmented_samples = parallel_map(apply_fn, samples,
                                         desc="Augmenting data", processes=num_processes, progress_bar=progress_bar)
        augmented_samples = sum(list(augmented_samples), [])  # Flatten
        return augmented_samples

    def apply(self, sample: Sample) -> Sample:
        # data augmentation
        if not self.params.data_aug_params.no_augs() \
                and sample.inputs is not None \
                and self.data_augmenter \
                and np.random.rand() <= self.params.data_aug_params.to_rel():
            line, text = self.augment(sample.inputs, sample.targets, sample.meta)
            return sample.new_inputs(line).new_targets(text)
        return sample

    def augment(self, line, text, meta):
        meta['augmented'] = True
        return self.data_augmenter.augment_single(line, text)

    def multi_augment(self, sample: Sample, n_augmentations=1, include_non_augmented=True):
        if include_non_augmented:
            out = [sample]
        else:
            out = []

        for n in range(n_augmentations):
            meta = copy.deepcopy(sample.meta)
            l, t = self.augment(sample.inputs, sample.targets, meta)
            out.append(Sample(inputs=l, targets=t, meta=meta))

        return out

