import numpy as np
from abc import ABC, abstractmethod

from calamari_ocr.proto import NetworkParams
from .ctc_decoder.default_ctc_decoder import DefaultCTCDecoder
from .ctc_decoder.fuzzy_ctc_decoder import FuzzyCTCDecoder
from calamari_ocr.ocr.datasets import InputDataset
from calamari_ocr.ocr import Codec

from typing import Any, Generator


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
    def __init__(self, network_proto, graph_type, batch_size, input_dataset: InputDataset = None, codec: Codec = None, processes=1):
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
        self.input_dataset = input_dataset
        self.codec = codec
        self.processes = processes

        self.ctc_decoder = {
            NetworkParams.CTC_FUZZY: FuzzyCTCDecoder(),
            NetworkParams.CTC_DEFAULT: DefaultCTCDecoder(),
        }[network_proto.ctc]

    def output_to_input_position(self, x):
        return x

    def set_input_dataset(self, input_dataset: InputDataset, codec: Codec):
        """ Set the networks data generator

        Parameters
        ----------
        data_generator : Generator[Tuple[np.array, np.array, Any], None, None]
            List of all raw labels to be used for training
        Returns
        -------
            None
        """
        self.input_dataset = input_dataset
        self.codec = codec

    def train_step(self):
        """ Performs a training step of the model.
        Returns
        -------
            None
        """

        return self.train()

    def iters_per_epoch(self, batch_size):
        size = len(self.input_dataset)
        r = size % batch_size
        n = size // batch_size
        return n if r == 0 else n + 1

    def predict_raw(self, x, len_x) -> Generator[NetworkPredictionResult, None, None]:
        pass

    def prediction_step(self) -> Generator[NetworkPredictionResult, None, None]:
        return self.predict()

    def reset_data(self):
        """ Called if the data changed
        """
        pass

    def prepare(self, uninitialized_variables_only=True, reset_queues=True):
        pass

    @abstractmethod
    def train(self):
        return []

    @abstractmethod
    def predict(self) -> Generator[NetworkPredictionResult, None, None]:
        """ Predict the current data

        Parameters
        ----------
        with_gt : bool
            Also output the gt if available in the dataset

        Returns
        -------
        list of Prediction

        See Also
        --------
            set_data
        """
        return []

    @abstractmethod
    def save_checkpoint(self, filepath):
        """ Save the current network state to `filepath`

        Parameters
        ----------
        filepath : str
            Where to store the checkpoint
        """
        pass

    @abstractmethod
    def load_weights(self, filepath, restore_only_trainable=True):
        """ Load the weights stored a the given `filepath`

        Parameters
        ----------
        filepath : str
            File to load
        restore_only_trainable : bool
            If False e.g. the solver state is loaded, which might not be desired.
        """
        pass

    @abstractmethod
    def realign_model_labels(self, indices_to_delete, indices_to_add):
        """ Realign the output matrix to the given labels

        On a codec resize some output labels can be added or deleted.
        Thus, the corresponding vectors in the output matrix of the DNN must be adapted accordingly.

        Parameters
        ----------
        indices_to_delete : list of int
            labels to be deleted
        indices_to_add : list of int
            labels to be added (usually at the end)

        """
        pass

