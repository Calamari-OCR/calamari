import os
import tempfile
import unittest

from tensorflow import keras

from calamari_ocr.ocr.dataset.data import Data
from calamari_ocr.ocr.dataset.datareader.file import FileDataParams
from calamari_ocr.ocr.dataset.imageprocessors.center_normalizer import CenterNormalizer
from calamari_ocr.ocr.dataset.imageprocessors.scale_to_height_processor import ScaleToHeight
from calamari_ocr.ocr.dataset.params import CalamariTrainOnlyGeneratorParams, CalamariSplitTrainValGeneratorParams
from calamari_ocr.ocr.scenario import Scenario
from calamari_ocr.scripts.train import main
from calamari_ocr.test.test_train_file import default_train_params, default_data_params
from calamari_ocr.utils import glob_all

this_dir = os.path.dirname(os.path.realpath(__file__))


class TestTrainFileNoPreload(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_simple_train(self):
        trainer_params = default_train_params(default_data_params(with_validation=False, preload=False))
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)

    def test_train_without_center_normalizer(self):
        trainer_params = default_train_params(default_data_params(with_validation=False, preload=False))
        trainer_params.scenario.data.pre_proc.replace_all(CenterNormalizer, ScaleToHeight())
        trainer_params.scenario.data.__post_init__()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)

    def test_train_with_val(self):
        trainer_params = default_train_params(default_data_params(with_validation=True, preload=False))
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)

    def test_train_split(self):
        trainer_params = default_train_params(default_data_params(with_split=True, preload=False))
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)


if __name__ == "__main__":
    unittest.main()
