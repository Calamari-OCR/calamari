import sys
import os
import tempfile
import unittest

import subprocess
from tensorflow import keras

from calamari_ocr.test.test_train_file import default_train_params, default_data_params
import calamari_ocr.scripts.train as train
import calamari_ocr.scripts.resume_training as resume_training


this_dir = os.path.dirname(os.path.realpath(__file__))


class TestTrainFile(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_simple_train(self):
        trainer_params = default_train_params(default_data_params(with_validation=False))
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            train.main(trainer_params)
            keras.backend.clear_session()
            resume_training.main([d])
