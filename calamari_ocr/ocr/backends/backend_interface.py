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
        self.implementation_handles_batching = False
        seed = network_proto.backend.random_seed
        if seed >= 0:
            random.seed(seed)
            np.random.seed(seed)

        super().__init__()
