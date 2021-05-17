import os
import unittest

from tfaip.data.databaseparams import DataPipelineParams

from calamari_ocr.ocr.predict.params import PredictorParams
from calamari_ocr.scripts.predict_and_eval import main, PredictAndEvalArgs

this_dir = os.path.dirname(os.path.realpath(__file__))


class TestPredictAndEval(unittest.TestCase):
    def test_predict_and_eval_uw3(self):
        from calamari_ocr.test.test_train_file import uw3_trainer_params

        checkpoint = os.path.join(this_dir, "models", "best.ckpt")
        trainer_params = uw3_trainer_params(with_validation=True)
        args = PredictAndEvalArgs(
            checkpoint=[checkpoint],
            predictor=PredictorParams(pipeline=DataPipelineParams(batch_size=5)),
            data=trainer_params.gen.val_gen(),
        )
        main(args)

    def test_predict_and_eval_page_xml(self):
        from calamari_ocr.test.test_train_pagexml import default_trainer_params

        checkpoint = os.path.join(this_dir, "models", "best.ckpt")
        trainer_params = default_trainer_params(with_validation=True)
        args = PredictAndEvalArgs(
            checkpoint=[checkpoint],
            predictor=PredictorParams(pipeline=DataPipelineParams(num_processes=1)),
            data=trainer_params.gen.val_gen(),
        )
        main(args)

    def test_predict_and_eval_abbyy(self):
        from calamari_ocr.test.test_train_abbyyxml import default_trainer_params

        checkpoint = os.path.join(this_dir, "models", "best.ckpt")
        trainer_params = default_trainer_params(with_validation=True)
        args = PredictAndEvalArgs(
            checkpoint=[checkpoint],
            predictor=PredictorParams(pipeline=DataPipelineParams(num_processes=1)),
            data=trainer_params.gen.val_gen(),
        )
        main(args)

    def test_predict_and_eval_hdf5(self):
        from calamari_ocr.test.test_train_hdf5 import default_trainer_params

        checkpoint = os.path.join(this_dir, "models", "best.ckpt")
        trainer_params = default_trainer_params(with_validation=True)
        args = PredictAndEvalArgs(
            checkpoint=[checkpoint],
            predictor=PredictorParams(pipeline=DataPipelineParams(num_processes=1)),
            data=trainer_params.gen.val_gen(),
        )
        main(args)

    def test_predict_and_eval_uw3_with_voting(self):
        from calamari_ocr.test.test_train_file import uw3_trainer_params

        checkpoint = os.path.join(this_dir, "models", "best.ckpt")
        trainer_params = uw3_trainer_params(with_validation=True)
        args = PredictAndEvalArgs(
            checkpoint=[checkpoint, checkpoint, checkpoint],
            predictor=PredictorParams(pipeline=DataPipelineParams(batch_size=5)),
            data=trainer_params.gen.val_gen(),
        )
        main(args)
