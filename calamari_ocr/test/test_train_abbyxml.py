import os
import tempfile
import unittest

from tensorflow import keras

from calamari_ocr.ocr.dataset.datareader.abbyy.reader import Abbyy
from calamari_ocr.ocr.scenario import CalamariScenario
from calamari_ocr.ocr.training.params import CalamariSplitTrainValGeneratorParams, CalamariTrainOnlyGeneratorParams
from calamari_ocr.scripts.train import main

this_dir = os.path.dirname(os.path.realpath(__file__))


def make_test_scenario(*, with_validation=False, with_split=False, preload=True):
    class CalamariPageXMLScenarioTest(CalamariScenario):
        @classmethod
        def default_trainer_params(cls):
            p = super().default_trainer_params()
            train = Abbyy(
                images=[
                    os.path.join(this_dir, "data", "hiltl_die_bank_des_verderbens_abbyyxml", "*.jpg"),
                ],
                preload=preload,
            )
            if with_split:
                p.gen = CalamariSplitTrainValGeneratorParams(validation_split_ratio=0.5, train=train)
            elif with_validation:
                p.gen.val = Abbyy(
                    images=[os.path.join(this_dir, "data", "hiltl_die_bank_des_verderbens_abbyyxml", "*.jpg")],
                    preload=preload
                )
                p.gen.train = train
                p.gen.__post_init__()
            else:
                p.gen = CalamariTrainOnlyGeneratorParams(train=train)

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

    return CalamariPageXMLScenarioTest


class TestAbbyyXML(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_simple_train(self):
        trainer_params = make_test_scenario(with_validation=False).default_trainer_params()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)

    def test_train_with_val(self):
        trainer_params = make_test_scenario(with_validation=True).default_trainer_params()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)

    def test_train_split(self):
        trainer_params = make_test_scenario(with_split=True).default_trainer_params()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)


class TestAbbyyXMLNoPreload(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_simple_train(self):
        trainer_params = make_test_scenario(with_validation=False, preload=False).default_trainer_params()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)

    def test_train_with_val(self):
        trainer_params = make_test_scenario(with_validation=True, preload=False).default_trainer_params()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)

    def test_train_split(self):
        trainer_params = make_test_scenario(with_split=True, preload=False).default_trainer_params()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)


if __name__ == "__main__":
    unittest.main()
