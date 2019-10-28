from abc import ABC, abstractmethod
import random
import numpy as np
from .model_interface import ModelInterface


class BackendInterface(ABC):
    def __init__(self,
                 checkpoint_params,
                 ):
        self.checkpoint_proto = checkpoint_params
        self.network_proto = self.checkpoint_proto.model.network
        seed = self.network_proto.backend.random_seed
        if seed >= 0:
            random.seed(seed)
            np.random.seed(seed)

        super().__init__()

    @abstractmethod
    def create_net(self, codec, graph_type, ctc_decoder_params=None, checkpoint_to_load=None, batch_size=-1, stream_input=True, codec_changes=None):
        pass
