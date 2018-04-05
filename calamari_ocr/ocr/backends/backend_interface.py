from abc import ABC, abstractmethod
import random
import numpy as np

class BackendInterface(ABC):
    def __init__(self,
                 network_proto,
                 ):
        self.network_proto = network_proto
        self.data = None
        self.labels = None
        self.implementation_handles_batching = False
        super().__init__()

    def set_data(self, data, labels=None):
        self.data = data
        self.labels = labels

        if labels is not None and len(data) != len(labels):
            raise Exception("Mismatch in size of data. Got {} images but {} labels".format(len(data), len(labels)))

    def train_step(self, batch_size):
        if self.implementation_handles_batching:
            batch_x = None
            batch_y = None
        else:
            indexes = [random.randint(0, len(self.data) - 1) for _ in range(batch_size)]
            batch_x = [self.data[i] for i in indexes]
            batch_y = [self.labels[i] for i in indexes]

        return self.train(batch_x, batch_y)

    def num_prediction_steps(self, batch_size):
        r = len(self.data) % batch_size
        n = len(self.data) // batch_size
        return n if r == 0 else n + 1

    def prediction_step(self, batch_size):
        for i in range(0, len(self.data), batch_size):
            batch_x = self.data[i:i + batch_size]
            for single in self.predict(batch_x):
                yield single

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


