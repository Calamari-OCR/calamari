import json
import time
from contextlib import ExitStack

from tensorflow import keras
from tfaip.base.device_config import DeviceConfig
from tfaip.base.predict import Predictor, PredictorParams, MultiModelPredictor
from tfaip.util.multiprocessing.parallelmap import tqdm_wrapper
from tqdm import tqdm

import numpy as np
from typing import Generator, List, Union

from calamari_ocr.ocr.backends.dataset.data_types import CalamariPipelineParams
from calamari_ocr.ocr.backends.scenario import CalamariScenario
from calamari_ocr.ocr.backends.tensorflow_backend.tensorflow_model import CalamariModel
from calamari_ocr.ocr.text_processing import text_processor_from_proto
from calamari_ocr.ocr import Codec, Checkpoint
from calamari_ocr.ocr.backends import create_backend_from_checkpoint
from calamari_ocr.utils.output_to_input_transformer import OutputToInputTransformer


class CalamariPredictorParams(PredictorParams):
    with_gt: bool = False
    ctc_decoder_params = None


class PredictionResult:
    def __init__(self, prediction, codec, text_postproc, out_to_in_trans, data_proc_params, ground_truth=None):
        """ The output of a networks prediction (PredictionProto) with additional information

        It stores all required information for decoding (`codec`) and interpreting the output.

        Parameters
        ----------
        prediction : PredictionProto
            prediction the DNN
        codec : Codec
            codec required to decode the `prediction`
        text_postproc : TextPostprocessor
            text processor to apply to the decodec `prediction` to receive the actual prediction sentence
        """
        self.prediction = prediction
        self.logits = np.reshape(prediction.logits.data, (prediction.logits.rows, prediction.logits.cols))
        self.codec = codec
        self.text_postproc = text_postproc
        self.chars = codec.decode(prediction.labels)
        self.sentence = self.text_postproc.apply("".join(self.chars))
        self.prediction.sentence = self.sentence
        self.out_to_in_trans = out_to_in_trans
        self.data_proc_params = data_proc_params
        self.ground_truth = ground_truth

        self.prediction.avg_char_probability = 0

        for p in self.prediction.positions:
            for c in p.chars:
                c.char = codec.code2char[c.label]

            p.global_start = int(self.out_to_in_trans.local_to_global(p.local_start, self.data_proc_params))
            p.global_end = int(self.out_to_in_trans.local_to_global(p.local_end, self.data_proc_params))
            if len(p.chars) > 0:
                self.prediction.avg_char_probability += p.chars[0].probability

        self.prediction.avg_char_probability /= len(self.prediction.positions) if len(
            self.prediction.positions) > 0 else 1


class CalamariPredictor(Predictor):
    def from_checkpoint(self, params: CalamariPredictorParams, checkpoint: str, auto_update_checkpoints=True):
        ckpt = Checkpoint(checkpoint, auto_update=auto_update_checkpoints)
        trainer_params = CalamariScenario.trainer_params_from_dict(ckpt.json)
        scenario = CalamariScenario(trainer_params.scenario_params)
        predictor = CalamariPredictor(params, scenario.create_data())
        scenario.setup_training('Adam')  # dummy setup
        model = scenario.keras_predict_model
        model.load_weights(ckpt.ckpt_path + 'h5')
        predictor.set_model(model)
        return predictor

    def __init__(self, params: CalamariPredictorParams, data: 'DataBase'):
        """ Predicting a dataset based on a trained model
        """
        super(CalamariPredictor, self).__init__(params, data)
        # TODO: transform: self.out_to_in_trans = OutputToInputTransformer(self.data_preproc, self.network)

    def predict_dataset(self, dataset, progress_bar=True, apply_preproc=True):
        """ Predict a complete dataset

        Parameters
        ----------
        dataset : Dataset
            Dataset to predict
        progress_bar : bool, optional
            hide or show a progress bar

        Yields
        -------
        PredictionResult
            Single PredictionResult
        dict
            Dataset entry of the prediction result
        """
        # if isinstance(dataset, RawDataSet):
        #     input_dataset = dataset.to_raw_input_dataset()
        # else:
        input_dataset = StreamingInputDataset(dataset, self.data_preproc if apply_preproc else None,
                                              self.text_postproc if apply_preproc else None)

        with input_dataset:
            prediction_results = self.predict_input_dataset(input_dataset, progress_bar)

            for prediction, sample in zip(prediction_results, dataset.samples()):
                yield prediction, sample

    def predict_raw(self, images, progress_bar=True, batch_size=-1, apply_preproc=True, params=None):
        batch_size = len(images) if batch_size < 0 else self.network.batch_size if batch_size == 0 else batch_size
        if apply_preproc:
            images, params = zip(*self.data_preproc.apply(images, progress_bar=progress_bar, processes=self.processes))

        for i in range(0, len(images), batch_size):
            input_images = images[i:i + batch_size]
            input_params = params[i:i + batch_size]
            for p, ip in zip(self.network.predict_raw(input_images), input_params):
                yield PredictionResult(p.decoded, codec=self.codec, text_postproc=self.text_postproc,
                                       out_to_in_trans=self.out_to_in_trans, data_proc_params=ip,
                                       ground_truth=p.ground_truth)

    def predict_input_dataset(self, input_dataset, progress_bar=True):
        """ Predict raw data
        Parameters
        ----------
        datas : list of array_like
            list of images
        progress_bar : bool, optional
            Show or hide a progress bar
        apply_preproc : bool, optional
            Apply the `data_preproc` to the `datas` before predicted by the DNN
        Yields
        -------
        PredictionResult
            A single PredictionResult
        """

        if progress_bar:
            out = tqdm(self.network.predict_dataset(input_dataset), desc="Prediction", total=len(input_dataset))
        else:
            out = self.network.predict_dataset(input_dataset)

        for p in out:
            yield PredictionResult(p.decoded, codec=self.codec, text_postproc=self.text_postproc,
                                   out_to_in_trans=self.out_to_in_trans, data_proc_params=p.params,
                                   ground_truth=p.ground_truth)


class MultiPredictor(ExitStack):
    def __init__(self, checkpoints: List[str] = None,
                 auto_update_checkpoints=True,
                 ):
        """Predict multiple models to use voting
        """
        super(MultiPredictor, self).__init__()
        checkpoints = checkpoints or []
        if len(checkpoints) == 0:
            raise Exception("No checkpoints provided.")

        predictor_params = PredictorParams(silent=True)
        DeviceConfig(predictor_params.device_params)
        self.checkpoints = [Checkpoint(ckpt, auto_update=auto_update_checkpoints) for ckpt in checkpoints]
        self.multi_predictor = MultiModelPredictor.from_paths([ckpt.json_path for ckpt in self.checkpoints],
                                                              predictor_params,
                                                              CalamariScenario,
                                                              model_paths=[ckpt.ckpt_path + '.h5' for ckpt in self.checkpoints],
                                                              )

    def __enter__(self):
        super(MultiPredictor, self).__enter__()
        self.enter_context(self.multi_predictor)
        return self

    def predict(self, dataset: CalamariPipelineParams, progress_bar=True):
        # TODO: compute correct dataset size
        for inputs, outputs, meta in tqdm_wrapper(self.multi_predictor.predict(dataset), progress_bar=progress_bar):
            yield outputs

        self.multi_predictor.benchmark_results.pretty_print()
