import os
import tempfile
import unittest

from tensorflow import keras

from calamari_ocr.ocr.dataset.datareader.file import FileDataParams
from calamari_ocr.ocr.scenario import CalamariEnsembleScenario
from calamari_ocr.scripts.train import main
from calamari_ocr.utils import glob_all

this_dir = os.path.dirname(os.path.realpath(__file__))


def setup_trainer_params(preload=True, debug=True):
    p = CalamariEnsembleScenario.default_trainer_params()
    p.scenario.debug_graph_construction = debug
    p.force_eager = debug

    p.gen.train = FileDataParams(
        images=glob_all([os.path.join(this_dir, "data", "uw3_50lines", "train", "*.png")]),
        preload=preload,
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


class TestTrainFile(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_simple_train_preload(self):
        trainer_params = setup_trainer_params(preload=True)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)

    def test_simple_train_no_preload(self):
        trainer_params = setup_trainer_params(preload=False)
        with tempfile.TemporaryDirectory() as d:
            trainer_params.output_dir = d
            main(trainer_params)


if __name__ == "__main__":
    unittest.main()
