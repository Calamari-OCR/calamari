from abc import ABC, abstractmethod
import codecs
import os
from enum import Enum
from typing import Tuple, Generator
from collections import namedtuple

import numpy as np

import multiprocessing as mp
import queue
from random import shuffle

from .datasetype import DataSetType
import logging
logger = logging.getLogger(__name__)


class DataSetMode(Enum):
    TRAIN = 0
    PREDICT = 1
    EVAL = 2
    PRED_AND_EVAL = 3


RequestParams = namedtuple('RequestParams', ('epochs', 'text_only'))


class DatasetGenerator:
    def __init__(self, mp_context, output_queue, mode, samples):
        self.output_queue = output_queue
        self.mode = mode
        self.samples = samples
        self.p = None
        self.request_queue = mp_context.Queue()

    def start(self):
        ctx = mp.get_context('spawn')
        self.p = ctx.Process(target=self.run, daemon=True)
        self.p.start()

    def stop(self):
        if self.p:
            self.p.terminate()
            self.p = None

    def join(self):
        if self.request_queue:
            self.request_queue.join()
            self.request_queue = None

        if self.p:
            self.p.join()

    def request(self, epochs, text_only=False):
        if not self.request_queue:
            raise Exception("Start not called yet.")
        self.request_queue.put(RequestParams(epochs, text_only))

    def run(self):
        global_index = 0
        while True:
            try:
                rq_params = self.request_queue.get(timeout=1)
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                return
            except Exception as e:
                logger.exception(e)
                return

            for epoch in range(rq_params.epochs):
                sample_idx = 0
                if self.mode == DataSetMode.TRAIN:
                    # no pred_and_eval bc it's shuffle
                    shuffle(self.samples)

                for sample in self.samples:
                    for line, text in self._load_sample(sample, rq_params.text_only):
                        while True:
                            try:
                                self.output_queue.put((global_index, sample_idx, line, text), timeout=0.1)
                            except queue.Full:
                                continue
                            except KeyboardInterrupt:
                                return

                            break

                        global_index += 1
                        sample_idx += 1

    @abstractmethod
    def _load_sample(self, sample, text_only) -> Generator[Tuple[np.array, str], None, None]:
        yield None, ""


class DataSet(ABC):
    def __init__(self, mode: DataSetMode, skip_invalid=False, remove_invalid=True):
        """ Dataset that stores a list of raw images and corresponding labels.

        Parameters
        ----------
        has_images : bool
            this dataset contains images
        has_texts : bool
            this dataset contains texts
        skip_invalid : bool
            skip invalid files instead of throwing an Exception
        remove_invalid : bool
            remove invalid files, thus dont count them to possible error on this data set
        """
        self._samples = []
        super().__init__()
        self.loaded = False
        self.mode = mode

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

    def store(self, extension):
        # either store text or store (e. g. if all predictions must be written at the same time
        pass

    @abstractmethod
    def create_generator(self, mp_context, output_queue) -> DatasetGenerator:
        return None


class RawDataSetGenerator(DatasetGenerator):
    def __init__(self, mp_context, output_queue, mode, samples):
        super().__init__(mp_context, output_queue, mode, samples)

    def _load_sample(self, sample, text_only) -> Generator[Tuple[np.array, str], None, None]:
        if text_only:
            yield None, sample['text']
        else:
            yield sample['image'], sample['text']


class RawDataSet(DataSet):
    def __init__(self, mode: DataSetMode, images=None, texts=None):
        """ Create a dataset from memory

        Since this dataset already contains all data in the memory, this dataset may not be loaded

        Parameters
        ----------
        images : list of images
            the images of the dataset
        texts : list of str
            the texts of this dataset
        """
        super().__init__(mode)

        if images is None and texts is None:
            raise Exception("Empty data set is not allowed. Both images and text files are None")

        if images is not None and texts is not None and len(images) == 0 and len(texts) == 0:
            raise Exception("Empty data set provided.")

        if texts is None or len(texts) == 0:
            if images is None:
                raise Exception("Empty data set.")

            # No gt provided, probably prediction
            texts = [None] * len(images)

        if images is None or len(images) == 0:
            if len(texts) is None:
                raise Exception("Empty data set.")

            # No images provided, probably evaluation
            images = [None] * len(texts)

        for i, (image, text) in enumerate(zip(images, texts)):
            self.add_sample({
                "image": image,
                "text": text,
                "id": str(i),
            })

        self.loaded = True

    def create_generator(self, mp_context, output_queue) -> DatasetGenerator:
        return RawDataSetGenerator(mp_context, output_queue, self.mode, self.samples())

