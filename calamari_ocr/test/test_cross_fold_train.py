import tempfile
import unittest

from tensorflow import keras

from calamari_ocr.ocr.training.cross_fold_trainer import CrossFoldTrainerParams
from calamari_ocr.scripts.cross_fold_train import main


def default_cross_fold_params(trainer_params):
    cfp = CrossFoldTrainerParams(
        trainer=trainer_params,
        n_folds=3,
    )
    return cfp


class TestCrossFoldTrain(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_on_files(self):
        from calamari_ocr.test.test_train_file import default_train_params, default_data_params
        trainer_params = default_train_params(default_data_params(with_validation=False, with_split=False))
        cfp = default_cross_fold_params(trainer_params)
        with tempfile.TemporaryDirectory() as d:
            cfp.best_models_dir = d
            main(cfp)

    def test_on_pagexml(self):
        from calamari_ocr.test.test_train_pagexml import default_train_params, default_data_params
        trainer_params = default_train_params(default_data_params(with_validation=False, with_split=False))
        cfp = default_cross_fold_params(trainer_params)
        with tempfile.TemporaryDirectory() as d:
            cfp.best_models_dir = d
            main(cfp)
