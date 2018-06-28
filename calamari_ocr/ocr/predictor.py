import time

from tqdm import tqdm

import numpy as np

from google.protobuf import json_format

from calamari_ocr.ocr.text_processing import text_processor_from_proto
from calamari_ocr.ocr.data_processing import data_processor_from_proto
from calamari_ocr.ocr import Codec
from calamari_ocr.ocr.backends import create_backend_from_proto
from calamari_ocr.proto import CheckpointParams
from calamari_ocr.proto import Prediction as PredictionProto
from calamari_ocr.proto import PredictionCharacter as PredictionCharProto
from calamari_ocr.proto import PredictionPosition as PredictionPosProto


class PredictionResult:
    def __init__(self, prediction, codec, text_postproc):
        self.prediction = prediction
        self.logits = np.reshape(prediction.logits.data, (prediction.logits.rows, prediction.logits.cols))
        self.codec = codec
        self.text_postproc = text_postproc
        self.chars = codec.decode(prediction.labels)
        self.sentence = self.text_postproc.apply("".join(self.chars))
        self.prediction.sentence = self.sentence

        for p in self.prediction.positions:
            for c in p.chars:
                c.char = codec.code2char[c.label]


class Predictor:
    def __init__(self, checkpoint=None, text_postproc=None, data_preproc=None, codec=None, network=None, batch_size=1, processes=1):
        self.network = network
        self.checkpoint = checkpoint
        self.codec = codec
        self.processes = processes

        if checkpoint:
            if network:
                raise Exception("Either a checkpoint or a network can be provided")

            with open(checkpoint + '.json', 'r') as f:
                checkpoint_params = json_format.Parse(f.read(), CheckpointParams())
                self.model_params = checkpoint_params.model

            self.network_params = self.model_params.network
            backend = create_backend_from_proto(self.network_params, restore=self.checkpoint, processes=processes)
            self.network = backend.create_net(restore=self.checkpoint, weights=None, graph_type="predict", batch_size=batch_size)
            self.text_postproc = text_postproc if text_postproc else text_processor_from_proto(self.model_params.text_postprocessor, "post")
            self.data_preproc = data_preproc if data_preproc else data_processor_from_proto(self.model_params.data_preprocessor)
        elif network:
            self.model_params = None
            self.network_params = network.network_proto
            self.text_postproc = text_postproc
            self.data_preproc = data_preproc
        else:
            raise Exception("Either a checkpoint or a existing backend must be provided")

    def predict_dataset(self, dataset, progress_bar=True):
        dataset.load_samples(processes=1, progress_bar=progress_bar)
        datas = dataset.prediction_samples()

        prediction_results = self.predict_raw(datas, progress_bar)

        for prediction, sample in zip(prediction_results, dataset.samples()):
            yield prediction, sample

    def predict_raw(self, datas, progress_bar=True, apply_preproc=True):
        # preprocessing step
        if apply_preproc:
            datas = self.data_preproc.apply(datas, processes=self.processes, progress_bar=progress_bar)

        codec = self.codec if self.codec else Codec(self.model_params.codec.charset)

        # create backend
        self.network.set_data(datas)

        if progress_bar:
            out = tqdm(self.network.prediction_step(), desc="Prediction", total=len(datas))
        else:
            out = self.network.prediction_step()

        for p in out:
            yield PredictionResult(p, codec=codec, text_postproc=self.text_postproc)


class MultiPredictor:
    def __init__(self, checkpoints=[], text_postproc=None, data_preproc=None, batch_size=1, processes=1):
        if len(checkpoints) == 0:
            raise Exception("No checkpoints provided.")

        self.processes = processes
        self.checkpoints = checkpoints
        self.predictors = [Predictor(cp, batch_size=batch_size, processes=processes) for cp in checkpoints]
        self.batch_size = batch_size

        # check if all checkpoints share the same preprocessor
        # then we only need to apply the preprocessing once and share the data accross the models
        preproc_params = self.predictors[0].model_params.data_preprocessor
        self.same_preproc = all([preproc_params == p.model_params.data_preprocessor for p in self.predictors])

    def predict_dataset(self, dataset, progress_bar=True):
        start_time = time.time()
        dataset.load_samples(processes=1, progress_bar=progress_bar)
        datas = dataset.prediction_samples()

        # preprocessing step (if all share the same preprocessor)
        if self.same_preproc:
            datas = self.predictors[0].data_preproc.apply(datas, processes=self.processes, progress_bar=progress_bar)

        def progress_bar_wrapper(l):
            if progress_bar:
                l = list(l)
                return tqdm(l, total=len(l), desc="Prediction")
            else:
                return l

        for data_idx in progress_bar_wrapper(range(0, len(datas), self.batch_size)):
            batch_data = datas[data_idx:data_idx+self.batch_size]
            samples = dataset.samples()[data_idx:data_idx+self.batch_size]

            # predict_raw returns list of [pred (batch_size), time]
            prediction = [predictor.predict_raw(batch_data, progress_bar=False, apply_preproc=not self.same_preproc)
                          for predictor in self.predictors]

            for result, sample in zip(zip(*prediction), samples):
                yield result, sample

        print("Prediction of {} models took {}s".format(len(self.predictors), time.time() - start_time))
