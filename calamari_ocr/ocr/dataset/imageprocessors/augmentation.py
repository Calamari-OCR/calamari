import copy
from dataclasses import dataclass, field
from functools import partial
from typing import List, Iterable, Type

import numpy as np
from dataclasses_json import config
from paiargparse import pai_dataclass, pai_meta
from tfaip.base.data.pipeline.definitions import Sample
from tfaip.base.data.pipeline.processor.dataprocessor import MappingDataProcessor, DataProcessorParams
from tfaip.util.multiprocessing.parallelmap import parallel_map

from calamari_ocr.ocr.augmentation import SimpleDataAugmenter
from calamari_ocr.ocr.augmentation.dataaugmentationparams import DataAugmentationAmount
from calamari_ocr.ocr.dataset.params import encoder, decoder


@pai_dataclass
@dataclass
class Augmentation(DataProcessorParams):
    data_aug_params: DataAugmentationAmount = field(
        default=DataAugmentationAmount.from_factor(0),
        metadata={**config(
            encoder=encoder,
            decoder=decoder(DataAugmentationAmount),
        ), **pai_meta(
            help="Amount of data augmentation per line (done before training). If this number is < 1 "
                 "the amount is relative.")
                  }
    )

    augmenter_type: str = 'simple'

    @staticmethod
    def cls() -> Type['MappingDataProcessor']:
        return Impl


class Impl(MappingDataProcessor[Augmentation]):
    def __init__(self, *args, **kwargs):
        super(Impl, self).__init__(*args, **kwargs)
        assert (self.params.augmenter_type == 'simple')
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
                                         desc="Augmenting data", processes=num_processes,
                                         progress_bar=progress_bar)
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
