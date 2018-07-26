import os
import json

from calamari_ocr.utils import glob_all


class CrossFold:
    def __init__(self, n_folds, source_files, output_dir):
        """ Prepare cross fold training

        This class creates folds out of the given source files.
        The individual splits are the optionally written to the `output_dir` in a json format.

        The file with index i will be assigned to fold i % n_folds (not randomly!)

        Parameters
        ----------
        n_folds : int
            the number of folds to create
        source_files : str
            the source file names
        output_dir : str
            where to store the folds
        """
        self.n_folds = n_folds
        self.inputs = sorted(glob_all(source_files))
        self.output_dir = os.path.abspath(output_dir)

        if len(self.inputs) == 0:
            raise Exception("No files found at '{}'".format(source_files))

        if self.n_folds <= 1:
            raise Exception("At least two folds are required")

        # fill single fold files
        self.folds = [[] for _ in range(self.n_folds)]
        for i, input in enumerate(self.inputs):
            self.folds[i % n_folds].append(input)

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
        }


        Parameters
        ----------
        filepath : str

        """
        with open(filepath, 'w') as f:
            json.dump({"folds": self.folds}, f, indent=4)


