import tempfile
import unittest

from tensorflow import keras

from calamari_ocr.ocr.augmentation.dataaugmentationparams import DataAugmentationAmount
from calamari_ocr.ocr.dataset.imageprocessors import Augmentation
from calamari_ocr.scripts.train import main
from calamari_ocr.test.test_train_file import default_train_params, default_data_params


class TestAugmentation(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_augmentation_train(self):
        trainer_params = default_train_params(default_data_params())
        trainer_params.scenario.data.pre_proc.processors_of_type(Augmentation)[
            0].data_aug_params = DataAugmentationAmount.from_factor(1)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)

    def test_augmentation_train_val(self):
        trainer_params = default_train_params(default_data_params(with_validation=True))
        trainer_params.scenario.data.pre_proc.processors_of_type(Augmentation)[
            0].data_aug_params = DataAugmentationAmount.from_factor(1)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)

    def test_augmentation_train_val_split(self):
        trainer_params = default_train_params(default_data_params(with_split=True))
        trainer_params.scenario.data.pre_proc.processors_of_type(Augmentation)[
            0].data_aug_params = DataAugmentationAmount.from_factor(1)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)


class TestAugmentationNoPreload(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_augmentation_train(self):
        trainer_params = default_train_params(default_data_params(preload=False))
        trainer_params.scenario.data.pre_proc.processors_of_type(Augmentation)[
            0].data_aug_params = DataAugmentationAmount.from_factor(1)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)

    def test_augmentation_train_val(self):
        trainer_params = default_train_params(default_data_params(with_validation=True, preload=False))
        trainer_params.scenario.data.pre_proc.processors_of_type(Augmentation)[
            0].data_aug_params = DataAugmentationAmount.from_factor(1)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)

    def test_augmentation_train_val_split(self):
        trainer_params = default_train_params(default_data_params(with_split=True, preload=False))
        trainer_params.scenario.data.pre_proc.processors_of_type(Augmentation)[
            0].data_aug_params = DataAugmentationAmount.from_factor(1)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.checkpoint_dir = d
            main(trainer_params)
