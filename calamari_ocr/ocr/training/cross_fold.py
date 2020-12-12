import os
import json
from contextlib import ExitStack
from typing import List

from tfaip.base.data.pipeline.definitions import Sample
from tfaip.util.multiprocessing.parallelmap import tqdm_wrapper

from calamari_ocr.ocr.dataset.datareader.file import FileDataReader
from calamari_ocr.ocr.dataset.datareader.hdf5 import Hdf5DatasetWriter
from calamari_ocr.ocr.dataset import DataSetType


class CrossFold:
    def __init__(self, n_folds: int, data_reader, output_dir: str, progress_bar=True,
                 ):
        """ Prepare cross fold training

        This class creates folds out of the given source files.
        The individual splits are the optionally written to the `output_dir` in a json format.

        The file with index i will be assigned to fold i % n_folds (not randomly!)
        """
        self.n_folds = n_folds
        self.data_reader = data_reader
        self.output_dir = os.path.abspath(output_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if len(self.data_reader) == 0:
            raise Exception("Empty dataset")

        if self.n_folds <= 1:
            raise Exception("At least two folds are required")

        # fill single fold files

        # if a FileDataSet, we can just use the paths of the images
        if isinstance(self.data_reader, FileDataReader):
            self.dataset_type = DataSetType.FILE
            self.folds = [[] for _ in range(self.n_folds)]
            for i, sample in enumerate(self.data_reader.samples()):
                self.folds[i % n_folds].append(sample['image_path'])
        else:
            self.dataset_type = DataSetType.HDF5
            # else load the data of each fold and write it to hd5 data files
            with ExitStack() as stack:
                folds = [stack.enter_context(Hdf5DatasetWriter(os.path.join(self.output_dir, 'fold{}'.format(i)))) for i in range(self.n_folds)]

                for i, sample in tqdm_wrapper(enumerate(self.data_reader.generate(epochs=1)), progress_bar=progress_bar,
                                              total=len(self.data_reader), desc="Creating hdf5 files"):
                    sample: Sample = sample
                    folds[i % self.n_folds].write(sample.inputs, sample.targets)

                self.folds = [f.files for f in folds]

    def train_files(self, fold: int) -> List[str]:
        """ List the train files of the `fold`
        """
        all_files = []
        for fold_id, inputs in enumerate(self.folds):
            if fold_id != fold:
                all_files += inputs

        return all_files

    def test_files(self, fold: int) -> List[str]:
        """ List the test files of the `fold`
        """
        for fold_id, inputs in enumerate(self.folds):
            if fold_id == fold:
                return inputs

        return []

    def write_folds_to_json(self, filepath: str):
        """ Write the fold split to the `filepath` as json.

        format is for 3 folds:
        {
            "folds": [
                [file1, file4, file7, ...],
                [file2, file5, file8, ...],
                [file3, file6, file9, ...]
            ]
            "type": FILE (or HDF5)
        }
        """
        with open(filepath, 'w') as f:
            json.dump({
                "type": self.dataset_type.name,
                "folds": self.folds,
            }, f, indent=4)


