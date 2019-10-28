import time

from tqdm import tqdm

import numpy as np
from typing import Generator, List

from calamari_ocr.ocr.text_processing import text_processor_from_proto
from calamari_ocr.ocr.data_processing import data_processor_from_proto
from calamari_ocr.ocr.datasets import InputDataset, StreamingInputDataset, RawInputDataset, RawDataSet
from calamari_ocr.ocr import Codec, Checkpoint
from calamari_ocr.ocr.backends import create_backend_from_checkpoint
from calamari_ocr.utils.output_to_input_transformer import OutputToInputTransformer
from contextlib import ExitStack


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

        self.prediction.avg_char_probability /= len(self.prediction.positions) if len(self.prediction.positions) > 0 else 1


class Predictor:
    def __init__(self, checkpoint=None, text_postproc=None, data_preproc=None, codec=None, network=None,
                 batch_size=1, processes=1,
                 auto_update_checkpoints=True,
                 with_gt=False,
                 ctc_decoder_params=None,
                 ):
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
        auto_update_checkpoints : bool, optional
            Update old models automatically (this will change the checkpoint files)
        with_gt : bool, optional
            The prediction will also output the ground truth if available else None
        ctc_decoder_params : optional
            Parameters of the ctc decoder
        """
        self.network = network
        self.checkpoint = checkpoint
        self.processes = processes
        self.auto_update_checkpoints = auto_update_checkpoints
        self.with_gt = with_gt

        if checkpoint:
            if network:
                raise Exception("Either a checkpoint or a network can be provided")

            ckpt = Checkpoint(checkpoint, auto_update=self.auto_update_checkpoints)
            self.checkpoint = ckpt.ckpt_path
            checkpoint_params = ckpt.checkpoint
            self.model_params = checkpoint_params.model
            self.codec = codec if codec else Codec(self.model_params.codec.charset)

            self.network_params = self.model_params.network
            backend = create_backend_from_checkpoint(checkpoint_params=checkpoint_params, processes=processes)
            self.text_postproc = text_postproc if text_postproc else text_processor_from_proto(self.model_params.text_postprocessor, "post")
            self.data_preproc = data_preproc if data_preproc else data_processor_from_proto(self.model_params.data_preprocessor)
            self.network = backend.create_net(
                codec=self.codec,
                ctc_decoder_params=ctc_decoder_params,
                checkpoint_to_load=ckpt,
                graph_type="predict", batch_size=batch_size)
        elif network:
            self.codec = codec
            self.model_params = None
            self.network_params = network.network_proto
            self.text_postproc = text_postproc
            self.data_preproc = data_preproc
            if not codec:
                raise Exception("A codec is required if preloaded network is used.")
        else:
            raise Exception("Either a checkpoint or a existing backend must be provided")

        self.out_to_in_trans = OutputToInputTransformer(self.data_preproc, self.network)

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
        if isinstance(dataset, RawDataSet):
            input_dataset = StreamingInputDataset(dataset, self.data_preproc if apply_preproc else None, self.text_postproc if apply_preproc else None)
        else:
            input_dataset = RawInputDataset(dataset, self.data_preproc if apply_preproc else None, self.text_postproc if apply_preproc else None)

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

    def predict_input_dataset(self, input_dataset: InputDataset, progress_bar=True):
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


class MultiPredictor:
    def __init__(self, checkpoints=None, text_postproc=None, data_preproc=None, batch_size=1, processes=1,
                 ctc_decoder_params=None,
                 ):
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
        ctc_decoder_params : optional
            Parameters of the ctc decoder
        """
        checkpoints = checkpoints if checkpoints else []
        if len(checkpoints) == 0:
            raise Exception("No checkpoints provided.")

        self.processes = processes
        self.checkpoints = checkpoints
        self.predictors = [Predictor(cp,
                                     ctc_decoder_params=ctc_decoder_params,
                                     batch_size=batch_size, text_postproc=text_postproc,
                                     data_preproc=data_preproc, processes=processes) for cp in checkpoints]
        self.batch_size = batch_size

        # check if all checkpoints share the same preprocessor
        # then we only need to apply the preprocessing once and share the data across the models
        preproc_params = self.predictors[0].model_params.data_preprocessor
        self.data_preproc = self.predictors[0].data_preproc
        self.same_preproc = all([preproc_params == p.model_params.data_preprocessor for p in self.predictors])

        # preprocessing step (if all share the same preprocessor)
        if not self.same_preproc:
            raise Exception('Different preprocessors are currently not allowed during prediction')

    def predict_raw(self, images, params=None, progress_bar=True, apply_preproc=True) -> Generator[List[PredictionResult], None, None]:
        if apply_preproc:
            images, params = zip(*self.data_preproc.apply(images, progress_bar=progress_bar))
        else:
            if not params:
                params = [None] * len(images)

        prediction = [predictor.predict_raw(images, progress_bar=progress_bar, apply_preproc=False, params=params) for predictor in self.predictors]

        for result in zip(*prediction):
            yield result

    def predict_dataset(self, dataset, progress_bar=True):
        start_time = time.time()
        with StreamingInputDataset(dataset, self.predictors[0].data_preproc, self.predictors[0].text_postproc, None,
                                   processes=self.processes,
                                   ) as input_dataset:
            def progress_bar_wrapper(l):
                if progress_bar:
                    return tqdm(l, total=int(np.ceil(len(dataset) / self.batch_size)), desc="Prediction")
                else:
                    return l

            def batched_data_params():
                batch = []
                for data_idx, (image, _, params) in enumerate(input_dataset.generator(epochs=1)):
                    batch.append((data_idx, image, params))
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []

                if len(batch) > 0:
                    yield batch

            for batch in progress_bar_wrapper(batched_data_params()):
                sample_ids, batch_images, batch_params = zip(*batch)
                samples = [dataset.samples()[i] for i in sample_ids]
                prediction = self.predict_raw(batch_images, params=batch_params, progress_bar=False, apply_preproc=False)
                for result, sample in zip(prediction, samples):
                    yield result, sample

        print("Prediction of {} models took {}s".format(len(self.predictors), time.time() - start_time))
