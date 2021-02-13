import os
import tempfile
import unittest

from tensorflow.python.keras.backend import clear_session

from calamari_ocr.scripts.train import main
from calamari_ocr.test.test_train_file import uw3_trainer_params

this_dir = os.path.dirname(os.path.realpath(__file__))


def default_trainer_params(*, preload=True):
    p = uw3_trainer_params(with_validation=True, preload=preload)
    p.warmstart.model = os.path.join(this_dir, 'models', 'best.ckpt.json')
    return p


class TestPretrained(unittest.TestCase):
    def tearDown(self) -> None:
        clear_session()

    def test_pretraining_with_codec_adaption_no_preload(self):
        trainer_params = default_trainer_params(preload=False)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_pretraining_with_codec_adaption(self):
        trainer_params = default_trainer_params()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_pretraining_without_codec_adaption(self):
        trainer_params = default_trainer_params()
        trainer_params.codec.keep_loaded = True
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)
