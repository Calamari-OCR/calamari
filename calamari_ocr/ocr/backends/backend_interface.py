from abc import ABC, abstractmethod
import random
import numpy as np
from .model_interface import ModelInterface


class BackendInterface(ABC):
    def __init__(self,
                 network_proto,
                 ):
        self.network_proto = network_proto
        self.implementation_handles_batching = False
        seed = network_proto.backend.random_seed
        if seed >= 0:
            random.seed(seed)
            np.random.seed(seed)

        super().__init__()

    @abstractmethod
    def create_net(self, dataset, codec, restore, weights, graph_type, batch_size=-1) -> ModelInterface:
        pass
