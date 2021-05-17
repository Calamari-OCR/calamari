import os
import tempfile
import unittest

from tensorflow import keras

from calamari_ocr.ocr.dataset.datareader.pagexml.reader import PageXML
from calamari_ocr.ocr.training.pipeline_params import (
    CalamariSplitTrainerPipelineParams,
    CalamariTrainOnlyPipelineParams,
)
from calamari_ocr.scripts.train import main
from calamari_ocr.test.calamari_test_scenario import CalamariTestScenario

this_dir = os.path.dirname(os.path.realpath(__file__))


def default_trainer_params(
    *,
    with_validation=False,
    with_split=False,
    preload=True,
    img_suffix="nrm.png",
    channels=1,
):
    p = CalamariTestScenario.default_trainer_params()
    train = PageXML(
        images=[
            os.path.join(this_dir, "data", "avicanon_pagexml", f"006.{img_suffix}"),
            os.path.join(this_dir, "data", "avicanon_pagexml", f"007.{img_suffix}"),
        ],
        preload=preload,
    )
    if with_split:
        p.gen = CalamariSplitTrainerPipelineParams(validation_split_ratio=0.5, train=train)
    elif with_validation:
        p.gen.val = PageXML(
            images=[os.path.join(this_dir, "data", "avicanon_pagexml", f"008.{img_suffix}")],
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
    p.scenario.data.pre_proc.run_parallel = False
    p.scenario.data.input_channels = channels
    p.scenario.data.__post_init__()
    p.scenario.__post_init__()
    p.__post_init__()
    return p


class TestPageXML(unittest.TestCase):
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

    def test_train_split(self):
        trainer_params = default_trainer_params(with_split=True)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)


class TestPageXMLNoPreload(unittest.TestCase):
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

    def test_train_split(self):
        trainer_params = default_trainer_params(with_split=True, preload=False)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)


if __name__ == "__main__":
    unittest.main()
