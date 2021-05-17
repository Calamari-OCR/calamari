import copy
from dataclasses import dataclass, field
from functools import partial
from typing import List, Iterable, Type

import numpy as np
from paiargparse import pai_dataclass, pai_meta
from tfaip.data.pipeline.definitions import Sample
from tfaip.data.pipeline.processor.dataprocessor import (
    MappingDataProcessor,
    DataProcessorParams,
)
from tfaip.util.multiprocessing.parallelmap import parallel_map

from calamari_ocr.ocr.augmentation.data_augmenter import (
    DataAugmenterParams,
    DefaultDataAugmenterParams,
)
from calamari_ocr.ocr.augmentation.dataaugmentationparams import DataAugmentationAmount


@pai_dataclass(alt="Augmentation")
@dataclass
class AugmentationProcessorParams(DataProcessorParams):
    augmenter: DataAugmenterParams = field(
        default_factory=DefaultDataAugmenterParams,
        metadata=pai_meta(
            mode="flat",
            help="Augmenter to use for augmentation",
            choices=[DefaultDataAugmenterParams],
        ),
    )
    n_augmentations: float = field(
        default=0,
        metadata=pai_meta(
            mode="flat",
            help="Amount of data augmentation per line (done before training). If this number is < 1 "
            "the amount is relative.",
        ),
    )

    @staticmethod
    def cls() -> Type["MappingDataProcessor"]:
        return AugmentationProcessor


class AugmentationProcessor(MappingDataProcessor[AugmentationProcessorParams]):
    def __init__(self, *args, **kwargs):
        super(AugmentationProcessor, self).__init__(*args, **kwargs)
        self.data_aug_params = DataAugmentationAmount.from_factor(self.params.n_augmentations)
        self.data_augmenter = self.params.augmenter.create()

    def preload(
        self,
        samples: List[Sample],
        num_processes=1,
        drop_invalid=True,
        progress_bar=False,
    ) -> Iterable[Sample]:
        n_augmentation = self.data_aug_params.to_abs()  # real number of augmentations
        if n_augmentation == 0:
            return samples

        apply_fn = partial(
            self.multi_augment,
            n_augmentations=n_augmentation,
            include_non_augmented=True,
        )
        augmented_samples = parallel_map(
            apply_fn,
            samples,
            desc="Augmenting data",
            processes=num_processes,
            progress_bar=progress_bar,
        )
        augmented_samples = sum(list(augmented_samples), [])  # Flatten
        return augmented_samples

    def apply(self, sample: Sample) -> Sample:
        # data augmentation
        if (
            not self.data_aug_params.no_augs()
            and sample.inputs is not None
            and self.data_augmenter
            and np.random.rand() <= self.data_aug_params.to_rel()
        ):
            line, text = self.augment(sample.inputs, sample.targets, sample.meta)
            return sample.new_inputs(line).new_targets(text)
        return sample

    def augment(self, line, text, meta):
        meta["augmented"] = True
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
