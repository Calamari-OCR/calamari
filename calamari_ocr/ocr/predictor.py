import time

from tqdm import tqdm

from google.protobuf import json_format

from calamari_ocr.ocr.text_processing import text_processor_from_proto
from calamari_ocr.ocr.data_processing import data_processor_from_proto
from calamari_ocr.ocr import Codec
from calamari_ocr.ocr.backends import create_backend_from_proto
from calamari_ocr.proto import CheckpointParams


class PredictionResult:
    def __init__(self, decoded, logits, codec, text_postproc):
        self.decoded = decoded
        self.logits = logits
        self.codec = codec
        self.text_postproc = text_postproc
        self.sentence = self.text_postproc.apply("".join(codec.decode(decoded)))


class Predictor:
    def __init__(self, checkpoint=None, text_postproc=None, data_preproc=None):
        if checkpoint:
            with open(checkpoint + '.json', 'r') as f:
                checkpoint_params = json_format.Parse(f.read(), CheckpointParams())
                self.model_params = checkpoint_params.model
        else:
            raise Exception("No checkpoint provided.")

        self.checkpoint = checkpoint
        self.text_postproc = text_postproc if text_postproc else text_processor_from_proto(self.model_params.text_postprocessor, "post")
        self.data_preproc = data_preproc if data_preproc else data_processor_from_proto(self.model_params.data_preprocessor)

    def predict_dataset(self, dataset, batch_size=1, processes=1, progress_bar=True):
        start_time = time.time()
        dataset.load_samples(processes=processes, progress_bar=progress_bar)
        datas = dataset.prediction_samples()

        prediction_results, prediction_time = self.predict_raw(datas, batch_size, processes, progress_bar)

        print("Total time: {}s, Prediction time: {}s, i. e. {}s per line.".format(
            time.time() - start_time, prediction_time,
            prediction_time / len(prediction_results)))

        return prediction_results, dataset.samples()

    def predict_raw(self, datas, batch_size=1, processes=1, progress_bar=True, apply_preproc=True):
        # preprocessing step
        if apply_preproc:
            datas = self.data_preproc.apply(datas, processes=processes, progress_bar=progress_bar)

        codec = Codec(self.model_params.codec.charset)

        # create backend
        network_params = self.model_params.network

        backend = create_backend_from_proto(network_params, restore=self.checkpoint)
        backend.set_prediction_data(datas)
        backend.prepare(train=False)

        prediction_start_time = time.time()

        if progress_bar:
            out = list(tqdm(backend.prediction_step(batch_size), desc="Prediction", total=backend.num_prediction_steps(batch_size)))
        else:
            out = list(backend.prediction_step(batch_size))

        prediction_results = [PredictionResult(
            decoded=d["decoded"],
            logits=d["logits"],
            codec=codec,
            text_postproc=self.text_postproc,
        ) for d in out]

        return prediction_results, time.time() - prediction_start_time


class MultiPredictor:
    def __init__(self, checkpoints=[], text_postproc=None, data_preproc=None):
        if len(checkpoints) == 0:
            raise Exception("No checkpoints provided.")

        self.checkpoints = checkpoints
        self.predictors = [Predictor(cp) for cp in checkpoints]

        # check if all checkpoints share the same preprocessor
        # then we only need to apply the preprocessing once and share the data accross the models
        preproc_params = self.predictors[0].model_params.data_preprocessor
        self.same_preproc = all([preproc_params == p.model_params.data_preprocessor for p in self.predictors])

    def predict_dataset(self, dataset, batch_size=1, processes=1, progress_bar=True):
        start_time = time.time()
        dataset.load_samples(processes=processes, progress_bar=progress_bar)
        datas = dataset.prediction_samples()

        # preprocessing step (if all share the same preprocessor)
        if self.same_preproc:
            datas = self.predictors[0].data_preproc.apply(datas, processes=processes, progress_bar=progress_bar)

        results = []
        for predictor in self.predictors:
            prediction_results, prediction_time =\
                predictor.predict_raw(datas, batch_size, processes, progress_bar,
                                      apply_preproc=not self.same_preproc)
            results.append(prediction_results)

        print("Prediction of {} models took {}s".format(len(self.predictors), time.time() - start_time))
        return results, dataset.samples()
