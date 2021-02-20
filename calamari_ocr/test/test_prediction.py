import unittest
import os
import numpy as np
from tensorflow import keras

from calamari_ocr.ocr import DataSetType
from calamari_ocr.ocr.dataset.datareader.file import FileDataParams
from calamari_ocr.ocr.predict.predictor import Predictor, PredictorParams, MultiPredictor
from calamari_ocr.utils import glob_all

from calamari_ocr.scripts.predict import run, PredictArgs
from calamari_ocr.utils.image import load_image

this_dir = os.path.dirname(os.path.realpath(__file__))


class PredictionAttrs:
    def __init__(self):
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


def file_dataset():
    return FileDataParams(
        images=sorted(glob_all([os.path.join(this_dir, "data", "uw3_50lines", "test", "*.png")]))
    )


def default_predictor_params():
    p = PredictorParams(
        progress_bar=False,
        silent=True
    )
    p.pipeline.batch_size = 2
    p.pipeline.num_processes = 1
    return p


def create_single_model_predictor():
    checkpoint = os.path.join(this_dir, "models", "0.ckpt")
    predictor = Predictor.from_checkpoint(default_predictor_params(), checkpoint=checkpoint)
    return predictor


def create_multi_model_predictor():
    checkpoint = os.path.join(this_dir, "models", "0.ckpt")
    predictor = MultiPredictor.from_paths(predictor_params=default_predictor_params(), checkpoints=[checkpoint, checkpoint])
    return predictor


def predict_args(n_models=1) -> PredictArgs:
    p = PredictArgs(
        checkpoint=[os.path.join(this_dir, "models", "0.ckpt")] * n_models,
        data=file_dataset(),
    )
    return p


class TestValidationTrain(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_prediction(self):
        run(predict_args())

    def test_prediction_voter(self):
        run(predict_args(n_models=3))

    def test_empty_image_raw_prediction(self):
        predictor = create_single_model_predictor()
        images = [np.zeros(shape=(0, 0)), np.zeros(shape=(1, 0)), np.zeros(shape=(0, 1))]
        for result in predictor.predict_raw(images):
            print(result.outputs.sentence)

    def test_white_image_raw_prediction(self):
        predictor = create_single_model_predictor()
        images = [np.zeros(shape=(200, 50))]
        for result in predictor.predict_raw(images):
            print(result.outputs.sentence)

    def test_raw_prediction(self):
        predictor = create_single_model_predictor()
        images = [load_image(file) for file in file_dataset().images]
        for result in predictor.predict_raw(images):
            self.assertGreater(result.outputs.avg_char_probability, 0)

    def test_dataset_prediction(self):
        predictor = create_single_model_predictor()
        for sample in predictor.predict(file_dataset()):
            self.assertGreater(sample.outputs.avg_char_probability, 0)

    def test_raw_prediction_voted(self):
        predictor = create_multi_model_predictor()
        images = [load_image(file) for file in file_dataset().images]
        for sample in predictor.predict_raw(images):
            r, voted = sample.outputs

    def test_dataset_prediction_voted(self):
        predictor = create_multi_model_predictor()
        for sample in predictor.predict(file_dataset()):
            r, voted = sample.outputs


if __name__ == "__main__":
    unittest.main()
