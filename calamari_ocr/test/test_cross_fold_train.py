import os
import tempfile
import unittest

from tensorflow import keras

from calamari_ocr.ocr.dataset.imageprocessors import AugmentationProcessorParams
from calamari_ocr.ocr.training.cross_fold_trainer import CrossFoldTrainerParams
from calamari_ocr.scripts.cross_fold_train import main
from calamari_ocr.test.test_train_file import (
    uw3_trainer_params as default_files_trainer_params,
)
from calamari_ocr.test.test_train_pagexml import (
    default_trainer_params as default_pagexml_trainer_params,
)


this_dir = os.path.dirname(os.path.realpath(__file__))


def default_cross_fold_params(trainer_params, pretrained="none", with_augmentation=False):
    cfp = CrossFoldTrainerParams(
        trainer=trainer_params,
        n_folds=3,
        max_parallel_models=1,
        single_fold=[0],  # Only one fold for tests
    )
    checkpoint = os.path.join(this_dir, "models", "best.ckpt")
    if pretrained == "one":
        cfp.weights = [checkpoint]
    elif pretrained == "all":
        cfp.weights = [checkpoint] * cfp.n_folds
    elif pretrained == "none":
        pass
    else:
        raise NotImplementedError

    if with_augmentation:
        for dp in cfp.trainer.scenario.data.pre_proc.processors_of_type(AugmentationProcessorParams):
            dp.n_augmentations = 1
    return cfp


class TestCrossFoldTrain(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_on_files(self):
        cfp = default_cross_fold_params(default_files_trainer_params())
        cfp.run_split = ""
        with tempfile.TemporaryDirectory() as d:
            cfp.best_models_dir = d
            main(cfp)

    def test_on_files_augmentation(self):
        cfp = default_cross_fold_params(default_files_trainer_params(), with_augmentation=True)
        with tempfile.TemporaryDirectory() as d:
            cfp.best_models_dir = d
            main(cfp)

    def test_on_files_one_pretrained(self):
        cfp = default_cross_fold_params(default_files_trainer_params(), pretrained="one")
        with tempfile.TemporaryDirectory() as d:
            cfp.best_models_dir = d
            main(cfp)

    def test_on_files_all_pretrained(self):
        cfp = default_cross_fold_params(default_files_trainer_params(), pretrained="all")
        with tempfile.TemporaryDirectory() as d:
            cfp.best_models_dir = d
            main(cfp)

    def test_on_files_no_preload(self):
        cfp = default_cross_fold_params(default_files_trainer_params(preload=False))
        with tempfile.TemporaryDirectory() as d:
            cfp.best_models_dir = d
            main(cfp)

    def test_on_pagexml(self):
        cfp = default_cross_fold_params(default_pagexml_trainer_params())
        with tempfile.TemporaryDirectory() as d:
            cfp.best_models_dir = d
            main(cfp)

    def test_on_pagexml_mixed_color_as_gray(self):
        cfp = default_cross_fold_params(default_pagexml_trainer_params(img_suffix="*.png"))
        with tempfile.TemporaryDirectory() as d:
            cfp.best_models_dir = d
            main(cfp)

    def test_on_pagexml_mixed_color_as_color(self):
        cfp = default_cross_fold_params(default_pagexml_trainer_params(img_suffix="*.png", channels=3))
        with tempfile.TemporaryDirectory() as d:
            cfp.best_models_dir = d
            main(cfp)
