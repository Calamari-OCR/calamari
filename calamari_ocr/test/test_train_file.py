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
from calamari_ocr.utils import glob_all

this_dir = os.path.dirname(os.path.realpath(__file__))


def default_data_params(with_validation=False, with_split=False, preload=True):
    train = FileDataParams(
        images=glob_all([os.path.join(this_dir, "data", "uw3_50lines", "train", "*.png")]),
        batch_size=1,
        num_processes=1,
        preload=preload,
    )
    params = Data.default_params()
    if with_split:
        params.gen = CalamariSplitTrainValGeneratorParams(validation_split_ratio=0.2, train=train)
    elif with_validation:
        params.val.images = glob_all([os.path.join(this_dir, "data", "uw3_50lines", "train", "*.png")])
        params.val.batch_size = 1
        params.val.num_processes = 1
        params.val.preload = preload
        params.gen.train = train
    else:
        params.gen = CalamariTrainOnlyGeneratorParams(train=train)
    params.__post_init__()
    return params


def default_train_params(data_params):
    params = Scenario.default_trainer_params()
    params.scenario.data = data_params
    params.epochs = 1
    params.samples_per_epoch = 2
    return params


class TestTrainFile(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_simple_train(self):
        trainer_params = default_train_params(default_data_params(with_validation=False))
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)

    def test_train_without_center_normalizer(self):
        trainer_params = default_train_params(default_data_params(with_validation=False))
        trainer_params.scenario.data.pre_proc.replace_all(CenterNormalizer, ScaleToHeight())
        trainer_params.scenario.data.__post_init__()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)

    def test_train_with_val(self):
        trainer_params = default_train_params(default_data_params(with_validation=True))
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)

    def test_train_split(self):
        trainer_params = default_train_params(default_data_params(with_split=True))
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)


if __name__ == "__main__":
    unittest.main()
