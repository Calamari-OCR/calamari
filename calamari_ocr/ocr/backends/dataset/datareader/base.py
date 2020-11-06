import codecs
import os
from abc import ABC, abstractmethod
from random import shuffle
from typing import Generator
import numpy as np

from calamari_ocr.ocr.backends.dataset.data_types import InputSample
from calamari_ocr.ocr.datasets.datasetype import DataSetMode


class DataReader(ABC):
    def __init__(self, mode: DataSetMode, skip_invalid=False, remove_invalid=True):
        """ Dataset that stores a list of raw images and corresponding labels.

        Parameters
        ----------
        skip_invalid : bool
            skip invalid files instead of throwing an Exception
        remove_invalid : bool
            remove invalid files, thus dont count them to possible error on this data set
        """
        self._samples = []
        self.loaded = False
        self.mode = mode
        self.auto_repeat = False

        self.skip_invalid = skip_invalid
        self.remove_invalid = remove_invalid

    def __len__(self):
        """ Number of samples

        Returns
        -------
        int
            Number of samples
        """
        return len(self._samples)

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

        self.loaded = False
        self._samples.append(sample)

    def is_sample_valid(self, sample, line, text):
        if self.mode == DataSetMode.PREDICT or self.mode == DataSetMode.TRAIN or self.mode == DataSetMode.PRED_AND_EVAL:
            # skip invalid imanges (e. g. corrupted or empty files)
            if line is None or (line.size == 0 or np.amax(line) == np.amin(line)):
                return False

        return True

    def store_text(self, sentence, sample, output_dir, extension):
        output_dir = output_dir if output_dir else os.path.dirname(sample['image_path'])
        with codecs.open(os.path.join(output_dir, sample['id'] + extension), 'w', 'utf-8') as f:
            f.write(sentence)

    def store_extended_prediction(self, data, sample, output_dir, extension):
        if extension == "pred":
            with open(os.path.join(output_dir, sample['id'] + ".pred"), 'wb') as f:
                f.write(data)
        elif extension == "json":
            with open(os.path.join(output_dir, sample['id'] + ".json"), 'w') as f:
                f.write(data)
        else:
            raise Exception("Unknown prediction format.")

    def prepare_store(self):
        pass

    def store(self, extension):
        # either store text or store (e. g. if all predictions must be written at the same time
        pass

    def generate(self, text_only=False) -> Generator[InputSample, None, None]:
        while True:
            if self.mode == DataSetMode.TRAIN:
                # no pred_and_eval bc it's shuffle
                shuffle(self._samples)

            for sample in self._samples:
                for raw_sample in self._load_sample(sample, text_only):
                    assert isinstance(raw_sample, InputSample)
                    yield raw_sample

            if not self.auto_repeat:
                break

    @abstractmethod
    def _load_sample(self, sample, text_only) -> Generator[InputSample, None, None]:
        raise NotImplementedError


