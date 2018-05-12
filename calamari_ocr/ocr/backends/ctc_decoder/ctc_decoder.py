from abc import ABC, abstractmethod


class CTCDecoder(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def decode(self, logits):
        pass
