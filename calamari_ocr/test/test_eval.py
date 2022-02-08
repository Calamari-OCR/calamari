import os
import tempfile
import unittest

from tensorflow import keras

from calamari_ocr.ocr import SavedCalamariModel
from calamari_ocr.ocr.dataset.datareader.abbyy.reader import Abbyy
from calamari_ocr.ocr.dataset.datareader.file import FileDataParams
from calamari_ocr.ocr.dataset.datareader.hdf5.reader import Hdf5
from calamari_ocr.ocr.dataset.datareader.pagexml.reader import PageXML
from calamari_ocr.scripts.predict import run as run_predict, PredictArgs
from calamari_ocr.scripts.eval import main as run_eval, EvalArgs
from calamari_ocr.utils import glob_all

this_dir = os.path.dirname(os.path.realpath(__file__))

checkpoint = os.path.join(this_dir, "models", f"version{SavedCalamariModel.VERSION}", "0.ckpt")


def predict_args(data) -> PredictArgs:
    p = PredictArgs(
        checkpoint=[checkpoint],
        data=data,
    )
    return p


def eval_args(gt_data, pred_data=None) -> EvalArgs:
    return EvalArgs(
        gt=gt_data,
        pred=pred_data,
        checkpoint=checkpoint,
    )


class TestEvaluation(unittest.TestCase):
    def tearDown(self) -> None:
        keras.backend.clear_session()

    def test_eval_files(self):
        run_predict(
            predict_args(
                data=FileDataParams(
                    images=sorted(glob_all([os.path.join(this_dir, "data", "uw3_50lines", "test", "*.png")]))
                )
            )
        )
        r = run_eval(
            eval_args(
                gt_data=FileDataParams(
                    texts=sorted(glob_all([os.path.join(this_dir, "data", "uw3_50lines", "test", "*.gt.txt")]))
                )
            )
        )
        self.assertLess(r["avg_ler"], 0.0009, msg="Current best model yields about 0.09% CER")
        args = eval_args(
            gt_data=FileDataParams(
                texts=sorted(glob_all([os.path.join(this_dir, "data", "uw3_50lines", "test", "*.gt.txt")]))
            )
        )
        with tempfile.TemporaryDirectory() as d:
            args.xlsx_output = os.path.join(d, "output.xlsx")
            run_eval(args)

    def test_eval_list_files(self):
        run_predict(
            predict_args(
                data=FileDataParams(
                    images=sorted(glob_all([os.path.join(this_dir, "data", "uw3_50lines", "test.files")]))
                )
            )
        )
        r = run_eval(
            eval_args(
                gt_data=FileDataParams(
                    texts=sorted(glob_all([os.path.join(this_dir, "data", "uw3_50lines", "test.gt.files")]))
                )
            )
        )
        self.assertLess(r["avg_ler"], 0.0009, msg="Current best model yields about 0.09% CER")

    def test_eval_files_with_different_extension(self):
        run_predict(
            predict_args(
                data=FileDataParams(
                    pred_extension=".ext-pred.txt",
                    images=sorted(glob_all([os.path.join(this_dir, "data", "uw3_50lines", "test", "*.png")])),
                )
            )
        )
        r = run_eval(
            eval_args(
                gt_data=FileDataParams(
                    pred_extension=".ext-pred.txt",
                    texts=sorted(glob_all([os.path.join(this_dir, "data", "uw3_50lines", "test", "*.gt.txt")])),
                )
            )
        )
        self.assertLess(r["avg_ler"], 0.0009, msg="Current best model yields about 0.09% CER")

    def test_eval_files_with_different_sources(self):
        run_predict(
            predict_args(
                data=FileDataParams(
                    pred_extension=".ext-pred.txt",
                    images=sorted(glob_all([os.path.join(this_dir, "data", "uw3_50lines", "test", "*.png")])),
                )
            )
        )
        r = run_eval(
            eval_args(
                gt_data=FileDataParams(
                    texts=sorted(glob_all([os.path.join(this_dir, "data", "uw3_50lines", "test", "*.gt.txt")]))
                ),
                pred_data=FileDataParams(
                    texts=sorted(
                        glob_all(
                            [
                                os.path.join(
                                    this_dir,
                                    "data",
                                    "uw3_50lines",
                                    "test",
                                    "*.ext-pred.txt",
                                )
                            ]
                        )
                    )
                ),
            )
        )
        self.assertLess(r["avg_ler"], 0.0009, msg="Current best model yields about 0.09% CER")

    def test_eval_pagexml(self):
        run_predict(
            predict_args(
                data=PageXML(
                    images=[os.path.join(this_dir, "data", "avicanon_pagexml", "008.nrm.png")],
                )
            )
        )
        run_eval(
            eval_args(
                gt_data=PageXML(
                    xml_files=[os.path.join(this_dir, "data", "avicanon_pagexml", "008.xml")],
                )
            )
        )

    def test_eval_abbyy(self):
        run_predict(
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
        run_eval(
            eval_args(
                gt_data=Abbyy(
                    xml_files=[
                        os.path.join(
                            this_dir,
                            "data",
                            "hiltl_die_bank_des_verderbens_abbyyxml",
                            "*.abbyy.xml",
                        )
                    ],
                )
            )
        )

    def test_eval_hdf5(self):
        run_predict(
            predict_args(
                data=Hdf5(
                    files=[os.path.join(this_dir, "data", "uw3_50lines", "uw3-50lines.h5")],
                )
            )
        )
        r = run_eval(
            eval_args(
                gt_data=Hdf5(
                    files=[os.path.join(this_dir, "data", "uw3_50lines", "uw3-50lines.h5")],
                )
            )
        )
        self.assertLess(r["avg_ler"], 0.006, msg="Current best model yields about 0.55% CER")
