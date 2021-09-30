import logging
import os
from abc import abstractmethod, ABC
from copy import deepcopy
from dataclasses import dataclass, field
from random import shuffle
from typing import Generator, Iterable, Optional, List, NoReturn, TypeVar

import numpy as np
from dataclasses_json import dataclass_json
from paiargparse import pai_dataclass, pai_meta
from tfaip import DataGeneratorParams
from tfaip.data.pipeline.datagenerator import DataGenerator
from tfaip.data.pipeline.definitions import PipelineMode, Sample

from calamari_ocr.utils.image import ImageLoaderParams, ImageLoader

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
            assert self.image.dtype == np.uint8

        if self.gt:
            assert type(self.gt) == str

        if self.meta:
            assert type(self.meta) == SampleMeta
        else:
            self.meta = SampleMeta(None)

    def to_input_target_sample(self) -> Sample:
        return Sample(inputs=self.image, targets=self.gt, meta=self.meta.to_dict())


@pai_dataclass
@dataclass
class CalamariDataGeneratorParams(DataGeneratorParams, ImageLoaderParams, ABC):
    skip_invalid: bool = True
    non_existing_as_empty: bool = False
    n_folds: int = field(default=-1, metadata=pai_meta(mode="ignore"))
    preload: bool = field(
        default=True,
        metadata=pai_meta(
            help="Instead of preloading all data, load the data on the fly. "
            "This is slower, but might be required for limited RAM or large dataset"
        ),
    )

    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def to_prediction(self):
        raise NotImplementedError

    def select(self, indices: List[int]):
        raise NotImplementedError

    def prepare_for_mode(self, mode: PipelineMode) -> NoReturn:
        pass

    def create(self, mode: PipelineMode) -> "CalamariDataGenerator":
        params = deepcopy(self)  # always copy of params
        params.prepare_for_mode(mode)
        gen: CalamariDataGenerator = self.cls()(mode, params)
        gen.post_init()
        return gen

    def image_loader(self) -> ImageLoader:
        return ImageLoader(self)


T = TypeVar("T", bound=CalamariDataGeneratorParams)


class CalamariDataGenerator(DataGenerator[T], ABC):
    def __init__(self, mode: PipelineMode, params: T):
        super(CalamariDataGenerator, self).__init__(mode, params)
        self._samples = []
        self._image_loader = params.image_loader()

    def _load_image(self, path: str) -> np.ndarray:
        return self._image_loader.load_image(path)

    def post_init(self):
        if self.params.n_folds > 0:
            logger.info(f"Populating {self.params.n_folds} folds")
            sample_idx = list(range(len(self._samples)))
            shuffle(sample_idx)
            for i, idx in enumerate(sample_idx):
                self._samples[i]["fold_id"] = i % self.params.n_folds

    def __len__(self):
        """Number of samples

        Returns
        -------
        int
            Number of samples
        """
        return len(self._samples)

    def sample_by_id(self, id_) -> dict:
        return next(sample for sample in self._samples if sample["id"] == id_)

    def samples(self) -> List[dict]:
        """List of all samples

        Returns
        -------
        list of dict
            List of all samples

        """
        return self._samples

    def add_sample(self, sample):
        """Add a sample

        Parameters
        ----------
        sample : dict
            The sample
        """
        if not isinstance(sample, dict):
            raise Exception("A sample is expected to be a dictionary")

        if "id" not in sample:
            raise Exception("A sample needs an id")

        if "fold_id" not in sample:
            sample["fold_id"] = -1  # dummy fold ID
        self._samples.append(sample)

    def store_text_prediction(self, prediction, sample_id, output_dir):
        raise NotImplementedError

    def store_extended_prediction(self, data, sample, output_dir, extension):
        bn = sample.get("base_name", sample["id"])
        if extension == "pred":
            with open(os.path.join(output_dir, bn + ".pred"), "wb") as f:
                f.write(data)
        elif extension == "json":
            with open(os.path.join(output_dir, bn + ".json"), "w") as f:
                f.write(data)
        else:
            raise Exception("Unknown prediction format.")

    def prepare_store(self):
        pass

    def store(self):
        # either store text or store (e. g. if all predictions must be written at the same time
        pass

    def generate(self) -> Iterable[Sample]:
        if self.mode == PipelineMode.TRAINING:
            # no pred_and_eval bc it's shuffle
            shuffle(self._samples)
        for sample in self._generate_epoch(text_only=self.mode == PipelineMode.TARGETS):
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
