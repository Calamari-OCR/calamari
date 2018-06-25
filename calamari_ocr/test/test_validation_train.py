import unittest
import os

from .test_simple_train import Attrs, this_dir, run, glob_all


class TestValidationTrain(unittest.TestCase):
    def test_validation_train(self):
        args = Attrs()
        args.validation = glob_all([os.path.join(this_dir, "data", "uw3_50lines", "test", "*.png")])
        args.max_iters = 30000

        run(args)

    def test_validation_pretrain(self):
        args = Attrs()
        args.validation = glob_all([os.path.join(this_dir, "data", "uw3_50lines", "test", "*.png")])
        args.max_iters = 1000
        args.early_stopping_best_model_prefix = args.early_stopping_best_model_prefix + "pretrain_"
        args.weights = os.path.join(this_dir, "test_models", "uw3_50lines_best.ckpt")

        run(args)


if __name__ == "__main__":
    unittest.main()
