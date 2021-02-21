import os
import tempfile
import unittest

from tensorflow.python.keras.backend import clear_session

from calamari_ocr.ocr.dataset.datareader.abbyy.reader import Abbyy
from calamari_ocr.ocr.dataset.datareader.pagexml.reader import PageXML
from calamari_ocr.ocr.scenario import CalamariScenario
from calamari_ocr.scripts.train import main

this_dir = os.path.dirname(os.path.realpath(__file__))


class CalamariAbbyyPageXMLScenarioTest(CalamariScenario):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        p.gen.train = Abbyy(
            images=[
                os.path.join(this_dir, "data", "hiltl_die_bank_des_verderbens_abbyyxml", "*.jpg"),
            ],
        )
        p.gen.val = PageXML(
            images=[os.path.join(this_dir, "data", "avicanon_pagexml", "008.nrm.png")],
        )

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


class TestTrainMixedData(unittest.TestCase):
    def tearDown(self) -> None:
        clear_session()

    def test_train_abbyy_test_page(self):
        trainer_params = CalamariAbbyyPageXMLScenarioTest.default_trainer_params()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)
