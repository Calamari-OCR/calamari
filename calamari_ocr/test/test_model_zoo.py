import os
import tempfile
import unittest
from glob import glob
from subprocess import check_call

import pytest
from tensorflow.python.keras.backend import clear_session
from tfaip.data.databaseparams import DataPipelineParams

from calamari_ocr.ocr.predict.params import PredictorParams
from calamari_ocr.scripts.predict_and_eval import (
    main as predict_and_eval_main,
    PredictAndEvalArgs,
)
from calamari_ocr.test.test_train_file import uw3_trainer_params


@pytest.mark.skipif(os.name != "posix", reason="Do not run on windows due to missing wget and untar.")
class TestModelZoo(unittest.TestCase):
    def tearDown(self) -> None:
        clear_session()

    def test_model_zoo(self):
        version = "2.0"
        url = f"https://github.com/Calamari-OCR/calamari_models/archive/{version}.tar.gz"
        with tempfile.TemporaryDirectory() as d:
            d = "model_archive_permanent"  # for debugging
            os.makedirs(d, exist_ok=True)
            os.chdir(d)
            if not os.path.exists("calamari_models"):
                check_call(
                    [
                        "sh",
                        "-c",
                        " ".join(
                            [
                                "wget",
                                "-q",
                                "-O",
                                "-",
                                url,
                                "|",
                                "tar",
                                "xz",
                                "&&",
                                "mv",
                                f"calamari_models-{version}",
                                "calamari_models",
                            ]
                        ),
                    ]
                )
            trainer_params = uw3_trainer_params(with_validation=True)
            args = PredictAndEvalArgs(
                checkpoint=glob(os.path.join("calamari_models", "uw3-modern-english", "*.ckpt.json")),
                predictor=PredictorParams(pipeline=DataPipelineParams(batch_size=5)),
                data=trainer_params.gen.val_gen(),
            )
            full_evaluation = predict_and_eval_main(args)
            self.assertLess(
                full_evaluation["voted"]["eval"]["avg_ler"],
                0.001,
                "The accuracy on the test data must be below 0.1%",
            )
