import os
import unittest

import numpy as np

from calamari_ocr.ocr import SavedCalamariModel
from calamari_ocr.ocr.predict.params import PredictionResult, Predictions
from tensorflow import keras

from calamari_ocr.ocr.dataset.datareader.abbyy.reader import Abbyy
from calamari_ocr.ocr.dataset.datareader.base import CalamariDataGeneratorParams
from calamari_ocr.ocr.dataset.datareader.extended_prediction import (
    ExtendedPredictionDataParams,
)
from calamari_ocr.ocr.dataset.datareader.file import FileDataParams
from calamari_ocr.ocr.dataset.datareader.hdf5.reader import Hdf5
from calamari_ocr.ocr.dataset.datareader.pagexml.reader import PageXML
from calamari_ocr.ocr.predict.predictor import (
    Predictor,
    PredictorParams,
    MultiPredictor,
)
from calamari_ocr.scripts.compute_average_prediction_confidence import (
    run as run_compute_avg_pred,
)
from calamari_ocr.scripts.predict import run, PredictArgs
from calamari_ocr.utils import glob_all
from calamari_ocr.utils.image import ImageLoaderParams

this_dir = os.path.dirname(os.path.realpath(__file__))
gray_scale_image_loader = ImageLoaderParams(channels=1).create()


def file_dataset():
    return FileDataParams(images=sorted(glob_all([os.path.join(this_dir, "data", "uw3_50lines", "test", "*.png")])))


def pagexml_dataset():
    return PageXML(
        images=[
            os.path.join(this_dir, "data", "avicanon_pagexml", f"006.nrm.png"),
        ],
    )


def default_predictor_params():
    p = PredictorParams(progress_bar=False, silent=True)
    p.pipeline.batch_size = 2
    p.pipeline.num_processes = 1
    return p


def create_single_model_predictor():
    checkpoint = os.path.join(this_dir, "models", "best.ckpt")
    predictor = Predictor.from_checkpoint(default_predictor_params(), checkpoint=checkpoint)
    return predictor


def create_multi_model_predictor():
    checkpoint = os.path.join(this_dir, "models", "best.ckpt")
    predictor = MultiPredictor.from_paths(
        predictor_params=default_predictor_params(),
        checkpoints=[checkpoint, checkpoint],
    )
    return predictor


def predict_args(n_models=1, data: CalamariDataGeneratorParams = file_dataset()) -> PredictArgs:
    p = PredictArgs(
        checkpoint=[os.path.join(this_dir, "models", "best.ckpt")] * n_models,
        data=data,
    )
    return p


class TestPrediction(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_prediction_files(self):
        run(predict_args())

    def test_prediction_extended_pagexml_with_voting(self):
        # With actual model to evaluate correct positions
        args = predict_args(data=pagexml_dataset())
        args.checkpoint = [os.path.join(this_dir, "models", f"version{SavedCalamariModel.VERSION}", "0.ckpt")] * 2
        args.extended_prediction_data = True
        run(args)
        jsons = [os.path.join(this_dir, "data", "uw3_50lines", "test", "*.json")]
        run_compute_avg_pred(ExtendedPredictionDataParams(files=jsons))

    def test_prediction_extended_and_positions(self):
        # With actual model to evaluate correct positions
        args = predict_args()
        args.checkpoint = [os.path.join(this_dir, "models", f"version{SavedCalamariModel.VERSION}", "0.ckpt")]
        args.extended_prediction_data = True
        run(args)
        jsons = [os.path.join(this_dir, "data", "uw3_50lines", "test", "*.json")]
        run_compute_avg_pred(ExtendedPredictionDataParams(files=jsons))

        def assert_pos_in_interval(p, start, end):
            self.assertGreaterEqual(p.global_start, start)
            self.assertGreaterEqual(p.global_end, start)
            self.assertLessEqual(p.global_start, end)
            self.assertLessEqual(p.global_end, end)

        with open(sorted(glob_all(jsons[0]))[0]) as f:
            first_pred: Predictions = Predictions.from_json(f.read())
            for p in first_pred.predictions:
                # Check for correct prediction string (models is trained!)
                self.assertEqual(p.sentence, "The problem, simplified for our purposes, is set up as")
                # Check for correct character positions
                assert_pos_in_interval(p.positions[0], 0, 24)  # T
                assert_pos_in_interval(p.positions[1], 24, 43)  # h
                assert_pos_in_interval(p.positions[2], 45, 63)  # e
                # ...
                assert_pos_in_interval(p.positions[-2], 1062, 1081)  # a
                assert_pos_in_interval(p.positions[-1], 1084, 1099)  # s

    def test_prediction_voter_files(self):
        run(predict_args(n_models=3))

    def test_prediction_pagexml(self):
        run(
            predict_args(
                data=PageXML(
                    images=[os.path.join(this_dir, "data", "avicanon_pagexml", "008.nrm.png")],
                )
            )
        )

    def test_prediction_abbyy(self):
        run(
            predict_args(
                data=Abbyy(
                    images=[
                        os.path.join(
                            this_dir,
                            "data",
                            "hiltl_die_bank_des_verderbens_abbyyxml",
                            "*.jpg",
                        )
                    ],
                )
            )
        )

    def test_prediction_hdf5(self):
        run(
            predict_args(
                data=Hdf5(
                    files=[os.path.join(this_dir, "data", "uw3_50lines", "uw3-50lines.h5")],
                )
            )
        )

    def test_empty_image_raw_prediction(self):
        predictor = create_single_model_predictor()
        images = [
            np.zeros(shape=(0, 0), dtype="float32"),
            np.zeros(shape=(1, 0), dtype="int16"),
            np.zeros(shape=(0, 1), dtype="uint8"),
        ]
        for result in predictor.predict_raw(images):
            print(result.outputs.sentence)

    def test_white_image_raw_prediction(self):
        predictor = create_single_model_predictor()
        images = [np.zeros(shape=(200, 50), dtype="uint8")]
        for result in predictor.predict_raw(images):
            print(result.outputs.sentence)

    def test_raw_prediction(self):
        predictor = create_single_model_predictor()
        images = [gray_scale_image_loader.load_image(file) for file in file_dataset().images] * 10
        for result in predictor.predict_raw(images):
            self.assertGreater(result.outputs.avg_char_probability, 0)

        predictor.benchmark_results.pretty_print()

    def test_raw_prediction_queue(self):
        predictor = create_single_model_predictor()
        images = [gray_scale_image_loader.load_image(file) for file in file_dataset().images]
        with predictor.raw() as raw_p:
            for image in images:
                r = raw_p(image)
                self.assertGreater(r.outputs.avg_char_probability, 0)

    def test_dataset_prediction(self):
        predictor = create_single_model_predictor()
        for sample in predictor.predict(file_dataset()):
            self.assertGreater(sample.outputs.avg_char_probability, 0)

    def test_raw_prediction_voted(self):
        predictor = create_multi_model_predictor()
        images = [gray_scale_image_loader.load_image(file) for file in file_dataset().images] * 2
        for sample in predictor.predict_raw(images):
            r, voted = sample.outputs

        predictor.benchmark_results.pretty_print()

    def test_dataset_prediction_voted(self):
        predictor = create_multi_model_predictor()
        for sample in predictor.predict(file_dataset()):
            r, voted = sample.outputs

        predictor.benchmark_results.pretty_print()


if __name__ == "__main__":
    unittest.main()
