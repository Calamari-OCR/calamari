import os
import tempfile
import unittest

from tensorflow import keras

from calamari_ocr.ocr.dataset.data import Data
from calamari_ocr.ocr.dataset.datareader.abbyy.reader import Abbyy
from calamari_ocr.ocr.dataset.params import CalamariTrainOnlyGeneratorParams, CalamariSplitTrainValGeneratorParams
from calamari_ocr.scripts.train import main
from calamari_ocr.test.test_train_file import default_train_params

this_dir = os.path.dirname(os.path.realpath(__file__))


def default_data_params(with_validation=False, with_split=False, preload=True):
    train = Abbyy(
        images=[
            os.path.join(this_dir, "data", "hiltl_die_bank_des_verderbens_abbyyxml", "*.jpg"),
        ],
        batch_size=1,
        num_processes=1,
        preload=preload,
    )
    params = Data.default_params()
    if with_split:
        params.gen = CalamariSplitTrainValGeneratorParams(validation_split_ratio=0.5, train=train)
    elif with_validation:
        params.gen.val = Abbyy(
            images=[os.path.join(this_dir, "data", "hiltl_die_bank_des_verderbens_abbyyxml", "*.jpg")],
            batch_size=1,
            num_processes=1,
            preload=preload
        )
        params.gen.train = train
    else:
        params.gen = CalamariTrainOnlyGeneratorParams(train=train)
    params.__post_init__()
    return params


class TestAbbyyXML(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_simple_train(self):
        trainer_params = default_train_params(default_data_params(with_validation=False))
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


class TestAbbyyXMLNoPreload(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_simple_train(self):
        trainer_params = default_train_params(default_data_params(with_validation=False, preload=False))
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
