from abc import ABC, abstractmethod
import codecs
import os
from enum import Enum

import numpy as np

from calamari_ocr.utils import parallel_map, split_all_ext


class DataSetMode(Enum):
    TRAIN = 0
    PREDICT = 1
    EVAL = 2


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

    def prediction_samples(self):
        """ Extract all images from this set

        Returns
        -------
        list of images

        """
        if not self.loaded:
            raise Exception("Dataset must be loaded to access its training samples")

        return [sample["image"] for sample in self._samples]

    def text_samples(self):
        """ Extract all texts from this set

        Returns
        -------
        list of str

        """
        if not self.loaded:
            raise Exception("Dataset must be loaded to access its text")

        return [sample["text"] for sample in self._samples]

    def train_samples(self, skip_empty=False):
        """ Extract both list of images and list of texts

        Parameters
        ----------
        skip_empty : bool
            do not add empty files

        Returns
        -------
        list of images
        list of str

        """
        if not self.loaded:
            raise Exception("Dataset must be loaded to access its training samples")

        data, text = [], []

        for sample in self._samples:
            if "text" not in sample:
                if skip_empty:
                    print("Skipping empty sample {}".format(sample["id"]))
                    continue

                raise Exception("Sample {} is not a train sample. "
                                "Maybe the corresponding txt file is missing".format(sample["id"]))

            data.append(sample["image"])
            text.append(sample["text"])

        return data, text

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

    def load_samples(self, processes=1, progress_bar=False):
        """ Load the samples into the memory

        This is usefull if a FileDataset shall load its files.

        Parameters
        ----------
        processes : int
            number of processes to use for loading
        progress_bar : bool
            show a progress bar of the progress

        Returns
        -------
        list of samples
        """
        if self.loaded:
            return self._samples

        data = parallel_map(self._load_sample, self._samples, desc="Loading Dataset", processes=processes, progress_bar=progress_bar)

        invalid_samples = []
        for i, ((line, text), sample) in enumerate(zip(data, self._samples)):
            sample["image"] = line
            sample["text"] = text
            if self.mode == DataSetMode.PREDICT or self.mode == DataSetMode.TRAIN:
                # skip invalid imanges (e. g. corrupted or empty files)
                if line is None or (line.size == 0 or np.amax(line) == np.amin(line)):
                    if self.skip_invalid:
                        invalid_samples.append(i)
                        if line is None:
                            print("Empty data: Image at '{}' is None (possibly corrupted)".format(sample['id']))
                        else:
                            print("Empty data: Image at '{}' is empty".format(sample['id']))
                    else:
                        raise Exception("Empty data: Image at '{}' is empty".format(sample['id']))

        if self.remove_invalid:
            # remove all invalid samples (reversed order!)
            for i in sorted(invalid_samples, reverse=True):
                del self._samples[i]

        self.loaded = True

        return self._samples

    @abstractmethod
    def _load_sample(self, sample):
        """ Load a single sample

        Parameters
        ----------
        sample : dict
            the sample to load

        Returns
        -------
        image
        text

        """
        return np.zeros((0, 0)), None

    def store_text(self, sentence, sample, output_dir, extension):
        output_dir = output_dir if output_dir else os.path.dirname(sample['image_path'])
        with codecs.open(os.path.join(output_dir, sample['id'] + extension), 'w', 'utf-8') as f:
            f.write(sentence)

    def store(self):
        # either store text or store (e. g. if all predictions must be written at the same time
        pass


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

    def _load_sample(self, sample):
        raise Exception("Raw dataset is always loaded")

