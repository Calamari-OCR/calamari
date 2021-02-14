import os
import tempfile
import unittest

from tensorflow import keras
from tfaip.base.data.pipeline.processor.params import SequentialProcessorPipelineParams

from calamari_ocr.ocr.dataset.data import Data
from calamari_ocr.ocr.dataset.datareader.hdf5.reader import Hdf5
from calamari_ocr.ocr.dataset.imageprocessors import PrepareSample
from calamari_ocr.ocr.dataset.params import CalamariTrainOnlyGeneratorParams
from calamari_ocr.scripts.train import main
from calamari_ocr.test.test_train_file import default_train_params

this_dir = os.path.dirname(os.path.realpath(__file__))


def default_data_params(with_validation=False, preload=True):
    train = Hdf5(
        files=[
            os.path.join(this_dir, "data", "uw3_50lines", "uw3-50lines.h5"),
        ],
        batch_size=1,
        num_processes=1,
        preload=preload,
    )
    params = Data.default_params()
    if with_validation:
        params.gen.val = Hdf5(
            files=[os.path.join(this_dir, "data", "uw3_50lines", "uw3-50lines.h5")],
            batch_size=1,
            num_processes=1,
            preload=preload
        )
        params.gen.train = train
    else:
        params.gen = CalamariTrainOnlyGeneratorParams(train=train)

    params.pre_proc = SequentialProcessorPipelineParams(
        processors=[PrepareSample()],
    )
    params.__post_init__()
    return params


class TestHDF5Train(unittest.TestCase):
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


class TestHDF5TrainNoPreload(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
