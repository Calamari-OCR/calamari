import os
import tempfile
import unittest

from tensorflow import keras
from tfaip.util.tfaipargparse import post_init

from calamari_ocr.ocr.dataset.datareader.file import FileDataParams
from calamari_ocr.ocr.scenario import CalamariEnsembleScenario
from calamari_ocr.scripts.train import main
from calamari_ocr.test.calamari_test_scenario import CalamariTestEnsembleScenario
from calamari_ocr.utils import glob_all

this_dir = os.path.dirname(os.path.realpath(__file__))


def setup_trainer_params(preload=True, debug=False):
    p = CalamariTestEnsembleScenario.default_trainer_params()
    p.force_eager = debug

    p.gen.train = FileDataParams(
        images=glob_all([os.path.join(this_dir, "data", "uw3_50lines", "train", "*.png")]),
        preload=preload,
    )

    post_init(p)
    return p


class TestTrainFile(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_simple_train_preload(self):
        trainer_params = setup_trainer_params(preload=True)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)


if __name__ == "__main__":
    unittest.main()
