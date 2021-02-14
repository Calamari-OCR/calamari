import logging
import os
from abc import abstractmethod, ABC
from copy import deepcopy
from dataclasses import dataclass, field
from random import shuffle
from typing import Generator, Iterable, Optional

import numpy as np
from dataclasses_json import dataclass_json
from paiargparse import pai_dataclass, pai_meta
from tfaip.base import DataGeneratorParams
from tfaip.base.data.pipeline.datagenerator import DataGenerator, T
from tfaip.base.data.pipeline.definitions import PipelineMode, Sample

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class SampleMeta:
    id: str
    augmented: bool = False
    fold_id: int = -1


@dataclass
class InputSample:
    image: Optional[np.ndarray]  # dtype uint8
    gt: Optional[str]
    meta: Optional[SampleMeta]

    def __post_init__(self):
        if self.image is not None:
            assert (self.image.dtype == np.uint8)

        if self.gt:
            assert (type(self.gt) == str)

        if self.meta:
            assert (type(self.meta) == SampleMeta)
        else:
            self.meta = SampleMeta(None)

    def to_input_target_sample(self) -> Sample:
        return Sample(inputs=self.image, targets=self.gt, meta=self.meta.to_dict())


@pai_dataclass
@dataclass
class CalamariDataGeneratorParams(DataGeneratorParams, ABC):
    skip_invalid: bool = False
    non_existing_as_empty: bool = False
    n_folds: int = field(default=-1, metadata=pai_meta(mode='ignore'))
    preload: bool = field(default=True, metadata=pai_meta(
        help='Instead of preloading all data, load the data on the fly. '
             'This is slower, but might be required for limited RAM or large dataset'
    ))

    def prepare_for_mode(self, mode: PipelineMode) -> DataGeneratorParams:
        pass

    def create(self, mode: PipelineMode) -> 'DataGenerator':
        params = deepcopy(self)  # always copy of params
        params.validate()
        params.prepare_for_mode(mode)
        gen: CalamariDataGenerator = self.cls()(mode, params)
        gen.post_init()
        return gen


class CalamariDataGenerator(DataGenerator[T], ABC):
    def __init__(self, mode: PipelineMode, params: T):
        super(CalamariDataGenerator, self).__init__(mode, params)
        self._samples = []

    def post_init(self):
        if self.params.n_folds > 0:
            logger.info(f"Populating {self.params.n_folds} folds")
            sample_idx = list(range(len(self._samples)))
            shuffle(sample_idx)
            for i, idx in enumerate(sample_idx):
                self._samples[i]['fold_id'] = i % self.params.n_folds

    def __len__(self):
        """ Number of samples

        Returns
        -------
        int
            Number of samples
        """
        return len(self._samples)

    def sample_by_id(self, id_) -> dict:
        return next(sample for sample in self._samples if sample['id'] == id_)

    def samples(self):
        """ List of all samples

        Returns
        -------
        list of dict
            List of all samples

        """
        return self._samples

    def add_sample(self, sample):
        """ Add a sample

        Parameters
        ----------
        sample : dict
            The sample
        """
        if not isinstance(sample, dict):
            raise Exception("A sample is expected to be a dictionary")

        if "id" not in sample:
            raise Exception("A sample needs an id")

        if 'fold_id' not in sample:
            sample['fold_id'] = -1  # dummy fold ID
        self._samples.append(sample)

    def store_text_prediction(self, sentence, sample_id, output_dir):
        raise NotImplementedError

    def store_extended_prediction(self, data, sample, output_dir, extension):
        bn = sample.get('base_name', sample['id'])
        if extension == "pred":
            with open(os.path.join(output_dir, bn + ".pred"), 'wb') as f:
                f.write(data)
        elif extension == "json":
            with open(os.path.join(output_dir, bn + ".json"), 'w') as f:
                f.write(data)
        else:
            raise Exception("Unknown prediction format.")

    def prepare_store(self):
        pass

    def store(self, extension):
        # either store text or store (e. g. if all predictions must be written at the same time
        pass

    def generate(self) -> Iterable[Sample]:
        if self.mode == PipelineMode.Training:
            # no pred_and_eval bc it's shuffle
            shuffle(self._samples)
        for sample in self._generate_epoch(text_only=self.mode == PipelineMode.Targets):
            yield sample.to_input_target_sample()

    def _generate_epoch(self, text_only) -> Generator[InputSample, None, None]:
        for sample in self._sample_iterator():
            for raw_sample in self._load_sample(sample, text_only=text_only):
                assert isinstance(raw_sample, InputSample)
                yield raw_sample

    def _sample_iterator(self):
        return self._samples

    @abstractmethod
    def _load_sample(self, sample, text_only) -> Generator[InputSample, None, None]:
        raise NotImplementedError
