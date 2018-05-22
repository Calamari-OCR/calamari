from abc import ABC, abstractmethod
import random
import numpy as np

from calamari_ocr.proto import LayerParams, NetworkParams
from .ctc_decoder.default_ctc_decoder import DefaultCTCDecoder
from .ctc_decoder.fuzzy_ctc_decoder import FuzzyCTCDecoder

class BackendInterface(ABC):
    def __init__(self,
                 network_proto,
                 ):
        self.network_proto = network_proto
        self.data_sets = {}
        self.implementation_handles_batching = False
        seed = network_proto.backend.random_seed
        if seed >= 0:
            random.seed(seed)
            np.random.seed(seed)

        self.ctc_decoder = {
            NetworkParams.CTC_FUZZY: FuzzyCTCDecoder(),
            NetworkParams.CTC_DEFAULT: DefaultCTCDecoder(),
        }[network_proto.ctc]

        super().__init__()

    def set_prediction_data(self, data):
        self.set_data("prediction", data)

    def set_train_data(self, data, labels):
        if len(data) != len(labels):
            raise Exception("Mismatch in size of data. Got {} images but {} labels".format(len(data), len(labels)))

        self.set_data("train", data, labels)

    def set_data(self, role, data, labels=None):
        self.data_sets[role] = {"data": data,
                                "labels": labels,
                                "indices": np.arange(0, len(data)),
                                "last_idx": len(data)}

        if labels is not None and len(data) != len(labels):
            raise Exception("Mismatch in size of data. Got {} images but {} labels".format(len(data), len(labels)))

    def train_step(self, batch_size, role="train"):
        if self.implementation_handles_batching:
            batch_x = None
            batch_y = None
        else:
            data_set = self.data_sets[role]
            data, labels = data_set["data"], data_set["labels"]
            indexes = [i for i in self.get_next_indices(data_set, batch_size)]
            batch_x = [data[i] for i in indexes]
            batch_y = [labels[i] for i in indexes]

        return self.train(batch_x, batch_y)

    def get_next_indices(self, data_set, total):
        last_idx = data_set["last_idx"]
        indices = data_set["indices"]
        for i in range(total):
            if last_idx >= len(indices):
                last_idx = 0
                np.random.shuffle(indices)

            yield indices[last_idx]
            last_idx += 1

        data_set["last_idx"] = last_idx

    def num_prediction_steps(self, batch_size, role="prediction"):
        data = self.data_sets[role]["data"]
        r = len(data) % batch_size
        n = len(data) // batch_size
        return n if r == 0 else n + 1

    def prediction_step(self, batch_size, role="prediction"):
        data = self.data_sets[role]["data"]
        for i in range(0, len(data), batch_size):
            batch_x = data[i:i + batch_size]
            for single in self.predict(batch_x):
                yield single

    @abstractmethod
    def realign_model_labels(self, indices_to_delete, indices_to_add):
        pass

    @abstractmethod
    def train(self, batch_x, batch_y):
        pass

    @abstractmethod
    def predict(self, batch_x):
        return []

    @abstractmethod
    def prepare(self, train):
        pass

    @abstractmethod
    def save_checkpoint(self, filepath):
        pass

    @abstractmethod
    def load_checkpoint_weights(self, filepath):
        pass


