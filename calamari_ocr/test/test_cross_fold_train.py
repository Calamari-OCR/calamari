import tempfile
import unittest

from tensorflow import keras

from calamari_ocr.ocr.training.cross_fold_trainer import CrossFoldTrainerParams
from calamari_ocr.scripts.cross_fold_train import main
from calamari_ocr.test.test_train_pagexml import make_test_scenario as make_hdf5_test_scenario
from calamari_ocr.test.test_train_file import make_test_scenario as make_files_test_scenario


class FilesTestScenario(make_files_test_scenario(with_validation=False, with_split=False)):
    pass

class FilesNoPreloadTestScenario(make_files_test_scenario(with_validation=False, with_split=False, preload=False)):
    pass

class Hdf5TestScenario(make_hdf5_test_scenario(with_validation=False, with_split=False)):
    pass


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
        cfp = default_cross_fold_params(FilesTestScenario.default_trainer_params())
        with tempfile.TemporaryDirectory() as d:
            cfp.best_models_dir = d
            main(cfp)

    def test_on_files_no_preload(self):
        cfp = default_cross_fold_params(FilesNoPreloadTestScenario.default_trainer_params())
        with tempfile.TemporaryDirectory() as d:
            cfp.best_models_dir = d
            main(cfp)

    def test_on_pagexml(self):
        cfp = default_cross_fold_params(Hdf5TestScenario.default_trainer_params())
        with tempfile.TemporaryDirectory() as d:
            cfp.best_models_dir = d
            main(cfp)
