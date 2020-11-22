import json
from functools import partial

from calamari_ocr.ocr.dataset.data import CalamariData
from calamari_ocr.ocr.voting import voter_from_params
from tfaip.base.data.pipeline.definitions import PipelineMode, InputOutputSample
from tfaip.base.device_config import DeviceConfig
from tfaip.base.predict import Predictor, PredictorParams, MultiModelPredictor
from tfaip.base.predict.multimodelpredictor import MultiModelVoter
from tqdm import tqdm

from typing import List, Callable

from calamari_ocr.ocr.dataset.params import CalamariPipelineParams
from calamari_ocr.ocr.scenario import CalamariScenario
from calamari_ocr.ocr import Codec, SavedModel
from calamari_ocr.utils.output_to_input_transformer import OutputToInputTransformer


class CalamariPredictorParams(PredictorParams):
    with_gt: bool = False
    ctc_decoder_params = None


class PredictionResult:
    def __init__(self, prediction, codec, text_postproc, out_to_in_trans: Callable[[int], int], ground_truth=None):
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
        self.logits = prediction.logits
        self.codec = codec
        self.text_postproc = text_postproc
        self.chars = codec.decode(prediction.labels)
        self.sentence = self.text_postproc.apply('', "".join(self.chars), {})[1]
        self.prediction.sentence = self.sentence
        self.out_to_in_trans = out_to_in_trans
        self.ground_truth = ground_truth

        self.prediction.avg_char_probability = 0

        for p in self.prediction.positions:
            for c in p.chars:
                c.char = codec.code2char[c.label]

            p.global_start = int(self.out_to_in_trans(p.local_start))
            p.global_end = int(self.out_to_in_trans(p.local_end))
            if len(p.chars) > 0:
                self.prediction.avg_char_probability += p.chars[0].probability

        self.prediction.avg_char_probability /= len(self.prediction.positions) if len(
            self.prediction.positions) > 0 else 1


class CalamariPredictor(Predictor):
    def from_checkpoint(self, params: CalamariPredictorParams, checkpoint: str, auto_update_checkpoints=True):
        ckpt = SavedModel(checkpoint, auto_update=auto_update_checkpoints)
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


class CalamariMultiModelVoter(MultiModelVoter):
    def __init__(self, voter_params, datas, post_proc, out_to_in_transformer):
        self.voter = voter_from_params(voter_params)
        self.datas = datas
        self.post_proc = post_proc
        self.out_to_in_transformer = out_to_in_transformer

    def vote(self, sample: InputOutputSample) -> InputOutputSample:
        inputs, outputs, meta = sample
        prediction_results = []
        input_meta = json.loads(inputs['meta'])
        for i, (prediction, m, data, post_) in enumerate(zip(outputs, meta, self.datas, self.post_proc)):
            prediction.id = "fold_{}".format(i)
            prediction_results.append(PredictionResult(prediction,
                                                       codec=data.params().codec,
                                                       text_postproc=post_,
                                                       out_to_in_trans=partial(self.out_to_in_transformer.local_to_global,
                                                                               model_factor=inputs['img_len'] /
                                                                                            prediction.logits.shape[0],
                                                                               data_proc_params=input_meta),
                                                       ))
        # vote the results (if only one model is given, this will just return the sentences)
        prediction = self.voter.vote_prediction_result(prediction_results)
        prediction.id = "voted"
        return InputOutputSample(inputs, (prediction_results, prediction), input_meta)


class MultiPredictor:
    def __init__(self, checkpoints: List[str] = None,
                 voter_params=None,
                 auto_update_checkpoints=True,
                 progress_bar=True,
                 ):
        """Predict multiple models to use voting
        """
        super(MultiPredictor, self).__init__()
        checkpoints = checkpoints or []
        if len(checkpoints) == 0:
            raise Exception("No checkpoints provided.")

        class CalamariMultiModelPredictor(MultiModelPredictor):
            def create_voter(self, data_params: 'DataBaseParams') -> MultiModelVoter:
                post_proc = [CalamariData.data_processor_factory().create_sequence(
                    data.params().post_processors_.sample_processors[1:], data.params(), PipelineMode.Prediction) for
                    data in self.datas]
                pre_proc = CalamariData.data_processor_factory().create_sequence(
                    self.data.params().pre_processors_.sample_processors, self.data.params(),
                    PipelineMode.Prediction)
                out_to_in_transformer = OutputToInputTransformer(pre_proc)
                return CalamariMultiModelVoter(voter_params, self.datas, post_proc, out_to_in_transformer)

        predictor_params = PredictorParams(silent=True, progress_bar=progress_bar)
        DeviceConfig(predictor_params.device_params)
        self.checkpoints = [SavedModel(ckpt, auto_update=auto_update_checkpoints) for ckpt in checkpoints]
        self.multi_predictor = CalamariMultiModelPredictor.from_paths([ckpt.json_path for ckpt in self.checkpoints],
                                                                      predictor_params,
                                                                      CalamariScenario,
                                                                      model_paths=[ckpt.ckpt_path + '.h5' for ckpt in self.checkpoints],
                                                                      )

    def data(self) -> CalamariData:
        return self.multi_predictor._data

    def predict(self, dataset: CalamariPipelineParams):
        for inputs, outputs, meta in self.multi_predictor.predict(dataset):
            yield inputs, outputs, meta

        self.multi_predictor.benchmark_results.pretty_print()
