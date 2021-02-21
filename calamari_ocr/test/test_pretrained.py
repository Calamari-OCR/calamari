import os
import tempfile
import unittest

from tensorflow.python.keras.backend import clear_session

from calamari_ocr.scripts.train import main
from calamari_ocr.test.test_train_file import make_test_scenario

this_dir = os.path.dirname(os.path.realpath(__file__))


def make_pretrained_test_scenario(*, preload=True):
    class PretrainedTestScenario(make_test_scenario(with_validation=True, preload=preload)):
        @classmethod
        def default_trainer_params(cls):
            p = super().default_trainer_params()
            p.warmstart.model = os.path.join(this_dir, 'models', '0.ckpt.json')
            return p
    return PretrainedTestScenario


class TestPretrained(unittest.TestCase):
    def tearDown(self) -> None:
        clear_session()

    def test_pretraining_with_codec_adaption_no_preload(self):
        trainer_params = make_pretrained_test_scenario(preload=False).default_trainer_params()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)

    def test_pretraining_with_codec_adaption(self):
        trainer_params = make_pretrained_test_scenario().default_trainer_params()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)

    def test_pretraining_without_codec_adaption(self):
        trainer_params = make_pretrained_test_scenario().default_trainer_params()
        trainer_params.codec.keep_loaded = True
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)
