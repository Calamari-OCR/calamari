import numpy as np
from abc import ABC, abstractmethod

from calamari_ocr.proto import LayerParams, NetworkParams
from .ctc_decoder.default_ctc_decoder import DefaultCTCDecoder
from .ctc_decoder.fuzzy_ctc_decoder import FuzzyCTCDecoder


class ModelInterface(ABC):
    def __init__(self, network_proto, graph_type, batch_size, implementation_handles_batching=False):
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
        implementation_handles_batching : bool
            True if the backend handles the data flow (queue, piping, ...)
        """
        self.network_proto = network_proto
        self.graph_type = graph_type
        self.batch_size = batch_size
        self.raw_images = []
        self.raw_labels = []
        self.implementation_handles_batching = implementation_handles_batching
        self.last_index = 0
        self.indices = []

        self.ctc_decoder = {
            NetworkParams.CTC_FUZZY: FuzzyCTCDecoder(),
            NetworkParams.CTC_DEFAULT: DefaultCTCDecoder(),
        }[network_proto.ctc]

    def output_to_input_position(self, x):
        return x

    def set_data(self, images, labels=None):
        """ Set the networks data (images, and labels)

        Set the data to be processed by the network on train or predict.
        Labels must be set if this model is used for training.
        Setting the data resets the internal state of any data prefetcher.

        Parameters
        ----------
        images : list(array_like)
            List of all raw images to be used for training or prediction
        labels : list(int)
            List of all raw labels to be used for training
        Returns
        -------
            None
        """
        self.raw_images = images
        self.raw_labels = labels if labels and len(labels) > 0 else [[] for _ in range(len(images))]
        self.indices = list(range(len(images)))

        self.reset_data()

    def train_step(self):
        """ Performs a training step of the model.

        If the actual implementation does not handle batching, this function will create batches on its own.
        Returns
        -------
            None
        """
        if self.implementation_handles_batching:
            batch_x = None
            batch_y = None
        else:
            indexes = []
            while len(indexes) != self.batch_size:
                i = self._next_index()
                if len(self.raw_labels[i]) == 0:
                    # skip empty labels
                    continue
                else:
                    indexes.append(i)

            batch_x = [self.raw_images[i].astype(np.float32) / 255.0 for i in indexes]
            batch_y = [self.raw_labels[i] for i in indexes]

        return self.train(batch_x, batch_y)

    def _next_index(self):
        if self.last_index >= len(self.indices):
            self.last_index = 0
            np.random.shuffle(self.indices)

        out = self.indices[self.last_index]
        self.last_index += 1
        return out

    def iters_per_epoch(self, batch_size):
        data = self.raw_images
        r = len(data) % batch_size
        n = len(data) // batch_size
        return n if r == 0 else n + 1

    def prediction_step(self):
        return self.predict()

    def reset_data(self):
        """ Called if the data changed
        """
        pass

    def prepare(self):
        self.last_index = len(self.indices)

    @abstractmethod
    def train(self, batch_x, batch_y):
        """ train on the given batch

        Parameters
        ----------
        batch_x : list of images
            the images
        batch_y : list of int
            the labels

        Returns
        -------

        """
        return []

    @abstractmethod
    def predict(self):
        """ Predict the current data

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

