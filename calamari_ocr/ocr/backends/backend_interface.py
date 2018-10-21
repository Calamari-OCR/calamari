from abc import ABC
import random
import numpy as np


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
