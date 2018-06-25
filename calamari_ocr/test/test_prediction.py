import unittest
import os

from calamari_ocr.utils import glob_all

from calamari_ocr.scripts.predict import run

this_dir = os.path.dirname(os.path.realpath(__file__))


class PredictionAttrs():
    def __init__(self):
        self.files = glob_all([os.path.join(this_dir, "data", "uw3_50lines", "test", "*.png")])
        self.checkpoint = [os.path.join(this_dir, "test_models", "uw3_50lines_best.ckpt")]
        self.processes = 1
        self.batch_size = 1
        self.verbose = True
        self.voter = "confidence_voter_default_ctc"
        self.output_dir = None
        self.extended_prediction_data = None
        self.extended_prediction_data_format = "json"
        self.no_progress_bars = True


class TestValidationTrain(unittest.TestCase):
    def test_prediction(self):
        args = PredictionAttrs()
        run(args)

    def test_prediction_voter(self):
        args = PredictionAttrs()
        self.checkpoint = [os.path.join(this_dir, "test_models", "uw3_50lines_best.ckpt")] * 2
        run(args)

if __name__ == "__main__":
    unittest.main()
