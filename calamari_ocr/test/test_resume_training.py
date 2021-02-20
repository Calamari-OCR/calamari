import os
import tempfile
import unittest

from tensorflow import keras

import calamari_ocr.scripts.resume_training as resume_training
import calamari_ocr.scripts.train as train
from calamari_ocr.test.test_train_file import make_test_scenario

this_dir = os.path.dirname(os.path.realpath(__file__))


class ResumeTrainingTestScenario(make_test_scenario(with_validation=False)):
    pass


class TestTrainFile(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_simple_train(self):
        trainer_params = ResumeTrainingTestScenario.default_trainer_params()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            train.main(trainer_params)
            keras.backend.clear_session()
            resume_training.main([os.path.join(d, 'checkpoint', 'checkpoint_0001')])
