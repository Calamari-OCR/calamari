import os
import tempfile
import unittest

from tensorflow import keras
from tfaip import INPUT_PROCESSOR

from calamari_ocr.ocr.dataset.datareader.file import FileDataParams
from calamari_ocr.ocr.dataset.imageprocessors.center_normalizer import CenterNormalizerProcessorParams
from calamari_ocr.ocr.dataset.imageprocessors.scale_to_height_processor import ScaleToHeightProcessorParams
from calamari_ocr.ocr.scenario import CalamariScenario
from calamari_ocr.ocr.training.pipeline_params import CalamariSplitTrainerPipelineParams, \
    CalamariTrainOnlyPipelineParams
from calamari_ocr.scripts.train import main
from calamari_ocr.utils import glob_all

this_dir = os.path.dirname(os.path.realpath(__file__))


def uw3_trainer_params(with_validation=False, with_split=False, preload=True, debug=False):
    p = CalamariScenario.default_trainer_params()
    p.scenario.debug_graph_construction = debug
    p.force_eager = debug

    train = FileDataParams(
        images=glob_all([os.path.join(this_dir, "data", "uw3_50lines", "train", "*.png")]),
        preload=preload,
    )
    if with_split:
        p.gen = CalamariSplitTrainerPipelineParams(validation_split_ratio=0.2, train=train)
    elif with_validation:
        p.gen.val.images = glob_all([os.path.join(this_dir, "data", "uw3_50lines", "test", "*.png")])
        p.gen.val.preload = preload
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
    p.scenario.data.pre_proc.run_parallel = False
    p.scenario.data.__post_init__()
    p.scenario.__post_init__()
    p.__post_init__()
    return p


class TestTrainFile(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_simple_train(self):
        trainer_params = uw3_trainer_params(with_validation=False)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_train_without_center_normalizer(self):
        trainer_params = uw3_trainer_params(with_validation=False)
        trainer_params.scenario.data.pre_proc.replace_all(CenterNormalizerProcessorParams,
                                                          ScaleToHeightProcessorParams())
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
        trainer_params.scenario.data.pre_proc.replace_all(CenterNormalizerProcessorParams,
                                                          ScaleToHeightProcessorParams(modes=INPUT_PROCESSOR))
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
