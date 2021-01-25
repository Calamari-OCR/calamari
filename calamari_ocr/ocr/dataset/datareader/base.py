import codecs
import os
from abc import abstractmethod, ABC
from random import shuffle
from typing import Generator
import numpy as np
from tfaip.base.data.pipeline.definitions import PipelineMode, Sample

from calamari_ocr.ocr.dataset.params import InputSample


class DataReader(ABC):
    def __init__(self, mode: PipelineMode, skip_invalid=False, remove_invalid=True, **kwargs):
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

        self.n_folds = -1

    def populate_folds(self, n_folds):
        self.n_folds = n_folds

        sample_idx = list(range(len(self._samples)))
        shuffle(sample_idx)
        for i, idx in enumerate(sample_idx):
            self._samples[i]['fold_id'] = i % n_folds

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

        self.loaded = False
        if 'fold_id' not in sample:
            sample['fold_id'] = -1  # dummy fold ID
        self._samples.append(sample)

    def is_sample_valid(self, sample, line, text):
        if self.mode == PipelineMode.Prediction or self.mode == PipelineMode.Training or self.mode == PipelineMode.Evaluation:
            # skip invalid imanges (e. g. corrupted or empty files)
            if line is None or (line.size == 0 or np.amax(line) == np.amin(line)):
                return False

        return True

    def store_text(self, sentence, sample, output_dir, extension):
        output_dir = output_dir if output_dir else os.path.dirname(sample['image_path'])
        bn = sample.get('base_name', sample['id'])
        with codecs.open(os.path.join(output_dir, bn + extension), 'w', 'utf-8') as f:
            f.write(sentence)

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

    def generate(self, epochs=1) -> Generator[Sample, None, None]:
        if self.auto_repeat:
            epochs = -1

        while epochs != 0:
            epochs -= 1
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


