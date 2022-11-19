import os
import shutil
import tempfile
import unittest

from tensorflow import keras
from tfaip import DeviceConfigParams
from tfaip.device.device_config import DeviceConfig

from calamari_ocr.ocr import SavedCalamariModel
from calamari_ocr.ocr.predict.predictor import Predictor
from calamari_ocr.test.test_prediction import default_predictor_params, file_dataset

this_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(this_dir, "models")


class TestModelMigration(unittest.TestCase):
    def tearDown(self):
        keras.backend.clear_session()

    def predict_and_eval(self, model_path):
        predictor = Predictor.from_checkpoint(default_predictor_params(), checkpoint=model_path)
        for sample in predictor.predict(file_dataset()):
            self.assertGreater(
                sample.outputs.avg_char_probability, 0.95
            )  # The model was trained and should yield good results

    def test_upgrade_from_5(self):
        with tempfile.TemporaryDirectory() as d:
            for filename in {"0.ckpt.h5", "0.ckpt.json"}:
                shutil.copyfile(
                    os.path.join(models_dir, "version5", filename),
                    os.path.join(d, filename),
                )

            ckpt = SavedCalamariModel(os.path.join(d, "0.ckpt.json"))
            self.predict_and_eval(ckpt.ckpt_path)

    def test_upgrade_from_4(self):
        with tempfile.TemporaryDirectory() as d:
            for filename in {"0.ckpt.h5", "0.ckpt.json"}:
                shutil.copyfile(
                    os.path.join(models_dir, "version4", filename),
                    os.path.join(d, filename),
                )

            ckpt = SavedCalamariModel(os.path.join(d, "0.ckpt.json"))
            self.predict_and_eval(ckpt.ckpt_path)

    def test_upgrade_from_3(self):
        with tempfile.TemporaryDirectory() as d:
            for filename in {"0.ckpt.h5", "0.ckpt.json"}:
                shutil.copyfile(
                    os.path.join(models_dir, "version3", filename),
                    os.path.join(d, filename),
                )

            ckpt = SavedCalamariModel(os.path.join(d, "0.ckpt.json"))
            self.predict_and_eval(ckpt.ckpt_path)
