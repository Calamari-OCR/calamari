import os
import json
from contextlib import ExitStack
from typing import List, Optional

from tfaip.data.pipeline.definitions import Sample, PipelineMode
from tfaip.util.multiprocessing.parallelmap import tqdm_wrapper

from calamari_ocr.ocr.dataset.datareader.base import CalamariDataGeneratorParams
from calamari_ocr.ocr.dataset.datareader.file import FileDataGenerator, FileDataParams
from calamari_ocr.ocr.dataset.datareader.hdf5 import Hdf5DatasetWriter


class CrossFold:
    def to_dict(self):
        return {
            "n_folds": self.n_folds,
            "data_generator_params": self.data_generator_params.to_dict(),
            "output_dir": self.output_dir,
            "folds": self.folds,
        }

    @staticmethod
    def from_dict(d):
        d["data_generator_params"] = CalamariDataGeneratorParams.from_dict(d["data_generator_params"])
        return CrossFold(**d)

    def __init__(
        self,
        n_folds: int,
        data_generator_params: CalamariDataGeneratorParams,
        output_dir: str,
        folds: Optional[List[str]] = None,
    ):
        """Prepare cross fold training

        This class creates folds out of the given source files.
        The individual splits are the optionally written to the `output_dir` in a json format.

        The file with index i will be assigned to fold i % n_folds (not randomly!)
        """
        self.n_folds = n_folds
        self.data_generator_params = data_generator_params
        self.output_dir = os.path.abspath(output_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if len(self.data_generator_params) == 0:
            raise Exception("Empty dataset")

        if self.n_folds <= 1:
            raise Exception("At least two folds are required")

        self.is_h5_dataset = not isinstance(self.data_generator_params, FileDataParams)
        self.folds = folds

    def create_folds(self, progress_bar):
        data_generator = self.data_generator_params.create(PipelineMode.EVALUATION)
        if len(data_generator) == 0:
            raise ValueError("Empty dataset.")
        if len(data_generator) < self.n_folds:
            raise ValueError(
                "Less files than folds in the dataset which results in folds without any training example."
            )

        if not self.is_h5_dataset:
            if isinstance(self.data_generator_params, FileDataParams):
                # if a FileDataSet, we can just use the paths of the images
                self.folds = [[] for _ in range(self.n_folds)]
                file_data_gen: FileDataGenerator = data_generator
                for i, sample in enumerate(file_data_gen.samples()):
                    self.folds[i % self.n_folds].append(sample["image_path"])
            else:
                raise NotImplementedError
        else:
            # else load the data of each fold and write it to hd5 data files
            with ExitStack() as stack:
                folds = [
                    stack.enter_context(Hdf5DatasetWriter(os.path.join(self.output_dir, "fold{}".format(i))))
                    for i in range(self.n_folds)
                ]
                for i, sample in tqdm_wrapper(
                    enumerate(data_generator.generate()),
                    progress_bar=progress_bar,
                    total=len(data_generator),
                    desc="Creating hdf5 files",
                ):
                    sample: Sample = sample
                    folds[i % self.n_folds].write(sample.inputs, sample.targets)

                self.folds = [f.files for f in folds]

    def train_files(self, fold: int) -> List[str]:
        """List the train files of the `fold`"""
        all_files = []
        for fold_id, inputs in enumerate(self.folds):
            if fold_id != fold:
                all_files += inputs

        return all_files

    def test_files(self, fold: int) -> List[str]:
        """List the test files of the `fold`"""
        for fold_id, inputs in enumerate(self.folds):
            if fold_id == fold:
                return inputs

        return []

    def write_folds_to_json(self, filepath: str):
        """Write the fold split to the `filepath` as json.

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
        with open(filepath, "w") as f:
            json.dump(
                {
                    "isH5": self.is_h5_dataset,
                    "folds": self.folds,
                },
                f,
                indent=4,
            )
