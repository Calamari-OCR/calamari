import os
import json
from contextlib import ExitStack

from calamari_ocr.utils import tqdm_wrapper
from calamari_ocr.ocr.datasets import DataSetType
from calamari_ocr.ocr.datasets.file_dataset import FileDataSet
from calamari_ocr.ocr.datasets.input_dataset import StreamingInputDataset
from calamari_ocr.ocr.data_processing import NoopDataPreprocessor
from calamari_ocr.ocr.text_processing import NoopTextProcessor
from calamari_ocr.ocr.datasets.hdf5_dataset.hdf5_dataset_writer import Hdf5DatasetWriter


class CrossFold:
    def __init__(self, n_folds, dataset, output_dir, progress_bar=True,
                 ):
        """ Prepare cross fold training

        This class creates folds out of the given source files.
        The individual splits are the optionally written to the `output_dir` in a json format.

        The file with index i will be assigned to fold i % n_folds (not randomly!)

        Parameters
        ----------
        n_folds : int
            the number of folds to create
        dataset : Dataset
            dataset containing all files
        output_dir : str
            where to store the folds
        """
        self.n_folds = n_folds
        self.dataset = dataset
        self.output_dir = os.path.abspath(output_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if len(self.dataset) == 0:
            raise Exception("Empty dataset")

        if self.n_folds <= 1:
            raise Exception("At least two folds are required")

        # fill single fold files

        # if a FileDataSet, we can just use the paths of the images
        if isinstance(self.dataset, FileDataSet):
            self.dataset_type = DataSetType.FILE
            self.folds = [[] for _ in range(self.n_folds)]
            for i, sample in enumerate(self.dataset.samples()):
                self.folds[i % n_folds].append(sample['image_path'])
        else:
            self.dataset_type = DataSetType.HDF5
            # else load the data of each fold and write it to hd5 data files
            with StreamingInputDataset(self.dataset, NoopDataPreprocessor(), NoopTextProcessor(), processes=1) as input_dataset:
                with ExitStack() as stack:
                    folds = [stack.enter_context(Hdf5DatasetWriter(os.path.join(self.output_dir, 'fold{}'.format(i)))) for i in range(self.n_folds)]

                    for i, (data, text, _) in tqdm_wrapper(enumerate(input_dataset.generator(epochs=1)), progress_bar=progress_bar,
                                                           total=len(dataset), desc="Creating hdf5 files"):
                        folds[i % self.n_folds].write(data, text)

                    self.folds = [f.files for f in folds]

    def train_files(self, fold):
        """ List the train files of the `fold`

        Parameters
        ----------
        fold : int
            index of the fold

        Returns
        -------
        list of str
            files in this fold
        See Also
        --------
        test_files
        """
        all_files = []
        for fold_id, inputs in enumerate(self.folds):
            if fold_id != fold:
                all_files += inputs

        return all_files

    def test_files(self, fold):
        """ List the test files of the `fold`

        Parameters
        ----------
        fold : int
            index of the fold

        Returns
        -------
        list of str
            files in this fold
        See Also
        --------
        train_files
        """
        for fold_id, inputs in enumerate(self.folds):
            if fold_id == fold:
                return inputs

        return []

    def write_folds_to_json(self, filepath):
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


        Parameters
        ----------
        filepath : str

        """
        with open(filepath, 'w') as f:
            json.dump({
                "type": self.dataset_type.name,
                "folds": self.folds,
            }, f, indent=4)


