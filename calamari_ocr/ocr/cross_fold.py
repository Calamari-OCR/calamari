import os
import json

from calamari_ocr.utils import glob_all


class CrossFold:
    def __init__(self, n_folds, source_files, output_dir):
        self.n_folds = n_folds
        self.inputs = glob_all(source_files)
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
        all_files = []
        for fold_id, inputs in enumerate(self.folds):
            if fold_id != fold:
                all_files += inputs

        return all_files

    def test_files(self, fold):
        for fold_id, inputs in enumerate(self.folds):
            if fold_id == fold:
                return inputs

        return []

    def write_folds_to_json(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({"folds": self.folds}, f, indent=4)


