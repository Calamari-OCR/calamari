import tempfile
import unittest

from tensorflow import keras

from calamari_ocr.ocr.dataset.imageprocessors import AugmentationProcessorParams
from calamari_ocr.scripts.train import main
from calamari_ocr.test.test_train_file import uw3_trainer_params


def default_trainer_params(*, with_validation=False, with_split=False, preload=True):
    p = uw3_trainer_params(with_validation=with_validation, with_split=with_split)
    if hasattr(p.gen, "val"):
        p.gen.val.preload = preload
    p.gen.train.preload = preload
    for dp in p.scenario.data.pre_proc.processors_of_type(AugmentationProcessorParams):
        dp.n_augmentations = 1
    return p


class TestAugmentation(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_augmentation_train(self):
        trainer_params = default_trainer_params(with_validation=False)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_augmentation_train_val(self):
        trainer_params = default_trainer_params(with_validation=True)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_augmentation_train_val_split(self):
        trainer_params = default_trainer_params(with_split=True)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)


class TestAugmentationNoPreload(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_augmentation_train(self):
        trainer_params = default_trainer_params(preload=False)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_augmentation_train_val(self):
        trainer_params = default_trainer_params(preload=False, with_validation=True)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_augmentation_train_val_split(self):
        trainer_params = default_trainer_params(preload=False, with_split=True)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)
