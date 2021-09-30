import os
import tempfile
import unittest

from tensorflow import keras
from tfaip import INPUT_PROCESSOR
from tfaip.util.tfaipargparse import post_init

from calamari_ocr.ocr.dataset.datareader.file import FileDataParams
from calamari_ocr.ocr.dataset.imageprocessors.center_normalizer import (
    CenterNormalizerProcessorParams,
)
from calamari_ocr.ocr.dataset.imageprocessors.scale_to_height_processor import (
    ScaleToHeightProcessorParams,
)
from calamari_ocr.ocr.training.pipeline_params import (
    CalamariSplitTrainerPipelineParams,
    CalamariTrainOnlyPipelineParams,
)
from calamari_ocr.scripts.train import main
from calamari_ocr.test.calamari_test_scenario import CalamariTestScenario
from calamari_ocr.utils import glob_all

this_dir = os.path.dirname(os.path.realpath(__file__))


def uw3_trainer_params(with_validation=False, with_split=False, preload=True, debug=False, from_files_file=False):
    p = CalamariTestScenario.default_trainer_params()
    p.force_eager = debug

    train_path = os.path.join(this_dir, "data", "uw3_50lines", "train", "*.png")
    test_path = os.path.join(this_dir, "data", "uw3_50lines", "test", "*.png")
    if from_files_file:
        train_path = os.path.join(this_dir, "data", "uw3_50lines", "train.files")
        test_path = os.path.join(this_dir, "data", "uw3_50lines", "test.files")

    train = FileDataParams(
        images=glob_all([train_path]),
        preload=preload,
    )
    if with_split:
        p.gen = CalamariSplitTrainerPipelineParams(validation_split_ratio=0.2, train=train)
    elif with_validation:
        p.gen.val.images = glob_all([test_path])
        p.gen.val.preload = preload
        p.gen.train = train
        p.gen.__post_init__()
    else:
        p.gen = CalamariTrainOnlyPipelineParams(train=train)

    p.gen.setup.val.batch_size = 1
    p.gen.setup.val.num_processes = 1
    p.gen.setup.train.batch_size = 1
    p.gen.setup.train.num_processes = 1
    post_init(p)
    return p


class TestTrainFile(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_simple_train(self):
        trainer_params = uw3_trainer_params(with_validation=False)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_train_from_files_file(self):
        trainer_params = uw3_trainer_params(with_validation=True, from_files_file=True)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_train_without_center_normalizer(self):
        trainer_params = uw3_trainer_params(with_validation=False)
        trainer_params.scenario.data.pre_proc.replace_all(
            CenterNormalizerProcessorParams, ScaleToHeightProcessorParams()
        )
        trainer_params.scenario.data.__post_init__()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_train_with_val(self):
        trainer_params = uw3_trainer_params(with_validation=True)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_train_split(self):
        trainer_params = uw3_trainer_params(with_split=True)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)


class TestTrainFileNoPreload(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_simple_train(self):
        trainer_params = uw3_trainer_params(with_validation=False, preload=False)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_train_without_center_normalizer(self):
        trainer_params = uw3_trainer_params(with_validation=False, preload=False)
        trainer_params.scenario.data.pre_proc.replace_all(
            CenterNormalizerProcessorParams,
            ScaleToHeightProcessorParams(modes=INPUT_PROCESSOR),
        )
        trainer_params.scenario.data.__post_init__()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_train_with_val(self):
        trainer_params = uw3_trainer_params(with_validation=True, preload=False)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_train_split(self):
        trainer_params = uw3_trainer_params(with_split=True, preload=False)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)


if __name__ == "__main__":
    unittest.main()
