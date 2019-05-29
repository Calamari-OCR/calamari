import unittest
import os
import numpy as np
from PIL import Image

from calamari_ocr.utils import glob_all

from calamari_ocr.scripts.predict import run
from calamari_ocr.ocr import DataSetType, Predictor, MultiPredictor

this_dir = os.path.dirname(os.path.realpath(__file__))


class PredictionAttrs:
    def __init__(self):
        self.files = sorted(glob_all([os.path.join(this_dir, "data", "uw3_50lines", "test", "*.png")]))
        self.checkpoint = [os.path.join(this_dir, "test_models", "uw3_50lines_best.ckpt")]
        # self.checkpoint = sorted(glob_all([os.path.join(this_dir, "..", "..", "..", "calamari_models/antiqua_modern/*.ckpt.json")]))
        self.processes = 1
        self.batch_size = 1
        self.verbose = True
        self.voter = "confidence_voter_default_ctc"
        self.output_dir = None
        self.extended_prediction_data = None
        self.extended_prediction_data_format = "json"
        self.no_progress_bars = True
        self.extension = None
        self.dataset = DataSetType.FILE
        self.text_files = None
        self.pagexml_text_index = 0


class TestValidationTrain(unittest.TestCase):
    def test_prediction(self):
        args = PredictionAttrs()
        args.checkpoint = args.checkpoint[0:1]
        run(args)

    def test_prediction_voter(self):
        args = PredictionAttrs()
        run(args)

    def test_raw_prediction(self):
        args = PredictionAttrs()
        predictor = Predictor(checkpoint=args.checkpoint[0])
        images = [np.array(Image.open(file), dtype=np.uint8) for file in args.files]
        for file, image in zip(args.files, images):
            r = list(predictor.predict_raw([image], progress_bar=False))[0]
            print(file, r.sentence)

    def test_raw_prediction_voted(self):
        args = PredictionAttrs()
        predictor = MultiPredictor(checkpoints=args.checkpoint)
        images = [np.array(Image.open(file), dtype=np.uint8) for file in args.files]
        for file, image in zip(args.files, images):
            r = list(predictor.predict_raw([image], progress_bar=False))[0]
            print(file, [rn.sentence for rn in r])


if __name__ == "__main__":
    unittest.main()
