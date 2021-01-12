import unittest
import os
import numpy as np
from PIL import Image
from tensorflow import keras

from calamari_ocr.ocr import DataSetType, PipelineParams
from calamari_ocr.ocr.predict.predictor import Predictor, PredictorParams, MultiPredictor
from calamari_ocr.utils import glob_all

from calamari_ocr.scripts.predict import run
from calamari_ocr.utils.image import load_image

this_dir = os.path.dirname(os.path.realpath(__file__))


class PredictionAttrs:
    def __init__(self):
        self.files = sorted(glob_all([os.path.join(this_dir, "data", "uw3_50lines", "test", "*.png")]))
        self.checkpoint = [os.path.join(this_dir, "models", "0.ckpt")]
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
        self.beam_width = 20
        self.dictionary = []
        self.dataset_pad = None


class TestValidationTrain(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_prediction(self):
        args = PredictionAttrs()
        args.checkpoint = args.checkpoint[0:1]
        run(args)

    def test_prediction_voter(self):
        args = PredictionAttrs()
        run(args)

    def test_raw_prediction(self):
        args = PredictionAttrs()
        predictor = Predictor.from_checkpoint(PredictorParams(progress_bar=False, silent=True), checkpoint=args.checkpoint[0])
        images = [load_image(file) for file in args.files]
        for file, image in zip(args.files, images):
            _, prediction, _ = list(predictor.predict_raw([image]))[0]
            print(file, prediction.sentence)

    def test_raw_dataset_prediction(self):
        args = PredictionAttrs()
        predictor = Predictor.from_checkpoint(PredictorParams(progress_bar=False, silent=True), checkpoint=args.checkpoint[0])
        params = PipelineParams(
            type=DataSetType.FILE,
            files=args.files
        )
        for inputs, outputs, meta in predictor.predict(params):
            pass

    def test_raw_prediction_voted(self):
        args = PredictionAttrs()
        predictor = MultiPredictor.from_paths(checkpoints=args.checkpoint, predictor_params=PredictorParams(progress_bar=False, silent=True))
        images = [load_image(file) for file in args.files]
        for inputs, (r, voted), meta in predictor.predict_raw(images):
            print([rn.sentence for rn in r])


if __name__ == "__main__":
    unittest.main()
