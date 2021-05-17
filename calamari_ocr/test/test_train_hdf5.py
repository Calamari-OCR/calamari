import os
import tempfile
import unittest

from tensorflow import keras
from tfaip.data.pipeline.processor.params import SequentialProcessorPipelineParams

from calamari_ocr.ocr.dataset.datareader.hdf5.reader import Hdf5
from calamari_ocr.ocr.dataset.imageprocessors import PrepareSampleProcessorParams
from calamari_ocr.ocr.training.pipeline_params import CalamariTrainOnlyPipelineParams
from calamari_ocr.scripts.train import main
from calamari_ocr.test.calamari_test_scenario import CalamariTestScenario

this_dir = os.path.dirname(os.path.realpath(__file__))


def default_trainer_params(with_validation=False, preload=True):
    p = CalamariTestScenario.default_trainer_params()
    train = Hdf5(
        files=[os.path.join(this_dir, "data", "uw3_50lines", "uw3-50lines.h5")],
        preload=preload,
    )
    if with_validation:
        p.gen.val = Hdf5(
            files=[os.path.join(this_dir, "data", "uw3_50lines", "uw3-50lines.h5")],
            preload=preload,
        )
        p.gen.train = train
        p.gen.__post_init__()
    else:
        p.gen = CalamariTrainOnlyPipelineParams(train=train)

    p.gen.setup.val.batch_size = 1
    p.gen.setup.val.num_processes = 1
    p.gen.setup.train.batch_size = 1
    p.gen.setup.train.num_processes = 1
    p.epochs = 1
    p.samples_per_epoch = 2
    p.scenario.data.__post_init__()
    p.scenario.__post_init__()
    p.__post_init__()
    return p


class TestHDF5Train(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_simple_train(self):
        trainer_params = default_trainer_params(with_validation=False)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_train_with_val(self):
        trainer_params = default_trainer_params(with_validation=True)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)


class TestHDF5TrainNoPreload(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_simple_train(self):
        trainer_params = default_trainer_params(with_validation=False, preload=False)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_train_with_val(self):
        trainer_params = default_trainer_params(with_validation=True, preload=False)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)


if __name__ == "__main__":
    unittest.main()
