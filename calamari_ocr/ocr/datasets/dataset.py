from abc import ABC, abstractmethod
import codecs
import os
from enum import Enum
from typing import Tuple, Generator

import numpy as np

from calamari_ocr.utils import parallel_map
from multiprocessing import Process, JoinableQueue
import multiprocessing as mp
import queue
from random import shuffle


class DataSetMode(Enum):
    TRAIN = 0
    PREDICT = 1
    EVAL = 2


class DatasetGenerator:
    def __init__(self, output_queue, mode, samples, text_only, epochs):
        self.output_queue = output_queue
        self.mode = mode
        self.epochs = epochs
        self.samples = samples
        self.text_only = text_only
        self.p = None

    def start(self):
        ctx = mp.get_context('spawn')
        self.p = ctx.Process(target=self.run, daemon=True)
        self.p.start()

    def join(self):
        if self.p:
            self.p.join()

    def run(self):
        global_index = 0
        for epoch in range(self.epochs):
            sample_idx = 0
            if self.mode == DataSetMode.TRAIN:
                shuffle(self.samples)

            for sample in self.samples:
                for line, text in self._load_sample(sample, self.text_only):
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
        if self.mode == DataSetMode.PREDICT or self.mode == DataSetMode.TRAIN:
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

    def store(self):
        # either store text or store (e. g. if all predictions must be written at the same time
        pass

    @abstractmethod
    def create_generator(self, output_queue, epochs, text_only) -> DatasetGenerator:
        return None


class RawDataSetGenerator(DatasetGenerator):
    def __init__(self, output_queue, mode, samples, text_only, epochs):
        super().__init__(output_queue, mode, samples, text_only, epochs)

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

    def create_generator(self, output_queue, epochs, text_only) -> DatasetGenerator:
        print("WARNING: RawData set should always be used with a RawInputDataSet to avoid excessive thread creation")
        return RawDataSetGenerator(output_queue, self.mode, self.samples(), text_only, epochs)

