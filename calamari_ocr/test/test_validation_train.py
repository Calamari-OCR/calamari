import tempfile
import unittest
import os

from tensorflow import keras

from calamari_ocr.test.test_simple_train import Attrs, this_dir, run, glob_all


class TestValidationTrain(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_validation_train(self):
        args = Attrs()
        args.validation = glob_all([os.path.join(this_dir, "data", "uw3_50lines", "test", "*.png")])
        args.validation_text_files = None
        with tempfile.TemporaryDirectory() as d:
            args.output_dir = d
            run(args)

    def test_validation_pretrain(self):
        args = Attrs()
        args.validation = glob_all([os.path.join(this_dir, "data", "uw3_50lines", "test", "*.png")])
        args.validation_text_files = None
        args.early_stopping_best_model_prefix = args.early_stopping_best_model_prefix + "pretrain_"
        args.weights = os.path.join(this_dir, "models", "0.ckpt")

        with tempfile.TemporaryDirectory() as d:
            args.output_dir = d
            run(args)


if __name__ == "__main__":
    unittest.main()
