import numpy as np
from abc import ABC, abstractmethod

from calamari_ocr.proto import LayerParams, NetworkParams
from .ctc_decoder.default_ctc_decoder import DefaultCTCDecoder
from .ctc_decoder.fuzzy_ctc_decoder import FuzzyCTCDecoder

class ModelInterface(ABC):
    def __init__(self, network_proto, graph_type, batch_size, implementation_handles_batching=False):
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

    def set_data(self, images, labels=None):
        self.raw_images = images
        self.raw_labels = labels if labels and len(labels) > 0 else [[] for _ in range(len(images))]
        self.indices = list(range(len(images)))

        self.reset_data()

    def train_step(self):
        if self.implementation_handles_batching:
            batch_x = None
            batch_y = None
        else:
            indexes = []
            while len(indexes) != self.batch_size:
                i = self.next_index()
                if len(self.raw_labels[i]) == 0:
                    # skip empty labels
                    continue
                else:
                    indexes.append(i)

            batch_x = [self.raw_images[i].astype(np.float32) / 255.0 for i in indexes]
            batch_y = [self.raw_labels[i] for i in indexes]

        return self.train(batch_x, batch_y)

    def next_index(self):
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
        pass

    def prepare(self):
        self.last_index = len(self.indices)

    @abstractmethod
    def train(self, batch_x, batch_y):
        return []

    @abstractmethod
    def predict(self):
        return []

    @abstractmethod
    def save_checkpoint(self, filepath):
        pass

    @abstractmethod
    def load_weights(self, filepath, restore_only_trainable=True):
        pass

    @abstractmethod
    def realign_model_labels(self, indices_to_delete, indices_to_add):
        pass

