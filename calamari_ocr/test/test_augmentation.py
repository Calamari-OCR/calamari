import tempfile
import unittest

from tensorflow import keras

from calamari_ocr.ocr.augmentation.dataaugmentationparams import DataAugmentationAmount
from calamari_ocr.ocr.dataset.imageprocessors import AugmentationProcessorParams
from calamari_ocr.scripts.train import main
from calamari_ocr.test.test_train_file import make_test_scenario


def make_aug_test_scenario(*, with_validation=False, with_split=False, preload=True):
    class AugmentationTestScenario(make_test_scenario(with_validation=with_validation, with_split=with_split)):
        @classmethod
        def default_trainer_params(cls):
            p = super().default_trainer_params()
            if hasattr(p.gen, 'val'):
                p.gen.val.preload = preload
            p.gen.train.preload = preload
            for dp in p.scenario.data.pre_proc.processors_of_type(AugmentationProcessorParams):
                dp.n_augmentations = 1
            return p

    return AugmentationTestScenario


class TestAugmentation(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_augmentation_train(self):
        trainer_params = make_aug_test_scenario(with_validation=False).default_trainer_params()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)

    def test_augmentation_train_val(self):
        trainer_params = make_aug_test_scenario(with_validation=True).default_trainer_params()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)

    def test_augmentation_train_val_split(self):
        trainer_params = make_aug_test_scenario(with_split=True).default_trainer_params()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)


class TestAugmentationNoPreload(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_augmentation_train(self):
        trainer_params = make_aug_test_scenario(preload=False).default_trainer_params()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)

    def test_augmentation_train_val(self):
        trainer_params = make_aug_test_scenario(preload=False, with_validation=True).default_trainer_params()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)

    def test_augmentation_train_val_split(self):
        trainer_params = make_aug_test_scenario(preload=False, with_split=True).default_trainer_params()
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)
