import time

from tqdm import tqdm

import numpy as np

from google.protobuf import json_format

from calamari_ocr.ocr.text_processing import text_processor_from_proto
from calamari_ocr.ocr.data_processing import data_processor_from_proto
from calamari_ocr.ocr import Codec
from calamari_ocr.ocr.backends import create_backend_from_proto
from calamari_ocr.proto import CheckpointParams
from calamari_ocr.utils.output_to_input_transformer import OutputToInputTransformer


class PredictionResult:
    def __init__(self, prediction, codec, text_postproc, out_to_in_trans, data_proc_params):
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

        for p in self.prediction.positions:
            for c in p.chars:
                c.char = codec.code2char[c.label]

            p.global_start = int(self.out_to_in_trans.local_to_global(p.local_start, self.data_proc_params))
            p.global_end = int(self.out_to_in_trans.local_to_global(p.local_end, self.data_proc_params))


class Predictor:
    def __init__(self, checkpoint=None, text_postproc=None, data_preproc=None, codec=None, network=None, batch_size=1, processes=1):
        """ Predicting a dataset based on a trained model

        Parameters
        ----------
        checkpoint : str, optional
            filepath of the checkpoint of the network to load, alternatively you can directly use a loaded `network`
        text_postproc : TextProcessor, optional
            text processor to be applied on the predicted sentence for the final output.
            If loaded from a checkpoint the text processor will be loaded from it.
        data_preproc : DataProcessor, optional
            data processor (must be the same as of the trained model) to be applied to the input image.
            If loaded from a checkpoint the text processor will be loaded from it.
        codec : Codec, optional
            Codec of the deep net to use for decoding. This parameter is only required if a custom codec is used,
            or a `network` has been provided instead of a `checkpoint`
        network : ModelInterface, optional
            DNN instance to used. Alternatively you can provide a `checkpoint` to load a network.
        batch_size : int, optional
            Batch size to use for prediction
        processes : int, optional
            The number of processes to use for prediction
        """
        self.network = network
        self.checkpoint = checkpoint
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
            if not codec:
                raise Exception("A codec is required if preloaded network is used.")
        else:
            raise Exception("Either a checkpoint or a existing backend must be provided")

        self.codec = codec if codec else Codec(self.model_params.codec.charset)
        self.out_to_in_trans = OutputToInputTransformer(self.data_preproc, self.network)

    def predict_dataset(self, dataset, progress_bar=True):
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
        dataset.load_samples(processes=1, progress_bar=progress_bar)
        datas = dataset.prediction_samples()
        data_params = zip(datas, [None] * len(datas))

        prediction_results = self.predict_raw(data_params, progress_bar)

        for prediction, sample in zip(prediction_results, dataset.samples()):
            yield prediction, sample

    def predict_raw(self, data_params, progress_bar=True, apply_preproc=True):
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

        datas, params = zip(*data_params)

        # preprocessing step
        if apply_preproc:
            datas, params = zip(*self.data_preproc.apply(datas, processes=self.processes, progress_bar=progress_bar))

        self.network.set_data(datas)

        if progress_bar:
            out = tqdm(self.network.prediction_step(), desc="Prediction", total=len(datas))
        else:
            out = self.network.prediction_step()

        for p, param in zip(out, params):
            yield PredictionResult(p, codec=self.codec, text_postproc=self.text_postproc, out_to_in_trans=self.out_to_in_trans, data_proc_params=param)


class MultiPredictor:
    def __init__(self, checkpoints=[], text_postproc=None, data_preproc=None, batch_size=1, processes=1):
        """Predict multiple models to use voting

        Parameters
        ----------
        checkpoints : list of str, optional
            list of the checkpoints to load
        text_postproc : TextProcessor, optional
            TextProcessor for the predicted sentence
        data_preproc : DataProcessor, optional
            DataProcessor for all input files
        batch_size : int, optional
            The number of files to process simultaneously by the DNN
        processes : int, optional
            The number of processes to use
        """
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
