import numpy as np
from abc import ABC, abstractmethod

from calamari_ocr.ocr.callbacks import TrainingCallback
from calamari_ocr.proto import NetworkParams
from .ctc_decoder.default_ctc_decoder import DefaultCTCDecoder
from .ctc_decoder.fuzzy_ctc_decoder import FuzzyCTCDecoder
from calamari_ocr.ocr.datasets import InputDataset
from calamari_ocr.ocr import Codec

from typing import Any, Generator, List


class NetworkPredictionResult:
    def __init__(self,
                 softmax: np.array,
                 output_length: int,
                 decoded: np.array,
                 params: Any = None,
                 ground_truth: np.array = None):
        self.softmax = softmax
        self.output_length = output_length
        self.decoded = decoded
        self.params = params
        self.ground_truth = ground_truth


class ModelInterface(ABC):
    def __init__(self, network_proto, graph_type, batch_size, codec: Codec = None,
                 processes=1):
        """ Interface for a neural net

        Interface above the actual DNN implementation to abstract training and prediction.

        Parameters
        ----------
        network_proto : NetworkParams
            Parameters that define the network
        graph_type : {"train", "test", "deploy"}
            Type of the graph, depending on the type different parts must be added (e.g. the solver)
        batch_size : int
            Number of examples to train/predict in parallel
        """
        self.network_proto = network_proto
        self.input_channels = network_proto.channels if network_proto.channels > 0 else 1
        self.graph_type = graph_type
        self.batch_size = batch_size
        self.codec = codec
        self.processes = processes

        self.ctc_decoder = DefaultCTCDecoder()

    def output_to_input_position(self, x):
        return x

    @abstractmethod
    def train(self, dataset, validation_dataset, checkpoint_params, text_post_proc, progress_bar,
              training_callback=TrainingCallback()):
        pass

    def iters_per_epoch(self, batch_size):
        size = len(self.input_dataset)
        r = size % batch_size
        n = size // batch_size
        return n if r == 0 else n + 1

    def predict_raw(self, x: List[np.array]) -> Generator[NetworkPredictionResult, None, None]:
        for r in self.predict_raw_batch(*self.zero_padding(x)):
            yield r

    @abstractmethod
    def predict_raw_batch(self, x: np.array, len_x: np.array) -> Generator[NetworkPredictionResult, None, None]:
        pass

    @abstractmethod
    def predict_dataset(self, dataset) -> Generator[NetworkPredictionResult, None, None]:
        """ Predict the current data

        Parameters
        ----------
        dataset : InputDataset
            the input dataset

        Returns
        -------
        list of Prediction

        See Also
        --------
            set_data
        """
        pass

    @abstractmethod
    def load_weights(self, filepath):
        """ Load the weights stored a the given `filepath`

        Parameters
        ----------
        filepath : str
            File to load
        """
        pass

    def zero_padding(self, data):
        len_x = [len(x) for x in data]
        out = np.zeros((len(data), max(len_x), self.network_proto.features), dtype=np.uint8)
        for i, x in enumerate(data):
            out[i, 0:len(x)] = x

        return np.expand_dims(out, axis=-1), np.array(len_x, dtype=np.int32)

