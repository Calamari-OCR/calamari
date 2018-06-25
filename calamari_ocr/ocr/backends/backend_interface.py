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

