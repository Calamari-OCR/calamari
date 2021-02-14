import os
import tempfile
import unittest

from tensorflow import keras

from calamari_ocr.ocr.dataset.data import Data
from calamari_ocr.ocr.dataset.datareader.pagexml.reader import PageXML
from calamari_ocr.ocr.dataset.params import CalamariTrainOnlyGeneratorParams, CalamariSplitTrainValGeneratorParams
from calamari_ocr.scripts.train import main
from calamari_ocr.test.test_train_file import default_train_params

this_dir = os.path.dirname(os.path.realpath(__file__))


def default_data_params(with_validation=False, with_split=False, preload=True):
    train = PageXML(
        images=[
            os.path.join(this_dir, "data", "avicanon_pagexml", "006.nrm.png"),
            os.path.join(this_dir, "data", "avicanon_pagexml", "007.nrm.png")
        ],
        batch_size=1,
        num_processes=1,
        preload=preload,
    )
    params = Data.default_params()
    if with_split:
        params.gen = CalamariSplitTrainValGeneratorParams(validation_split_ratio=0.5, train=train)
    elif with_validation:
        params.gen.val = PageXML(
            images=[os.path.join(this_dir, "data", "avicanon_pagexml", "008.nrm.png")],
            batch_size=1,
            num_processes=1,
            preload=preload
        )
        params.gen.train = train
    else:
        params.gen = CalamariTrainOnlyGeneratorParams(train=train)
    params.__post_init__()
    return params


class TestPageXML(unittest.TestCase):
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


class TestPageXMLNoPreload(unittest.TestCase):
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
