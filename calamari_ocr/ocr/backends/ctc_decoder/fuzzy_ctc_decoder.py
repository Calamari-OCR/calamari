from .ctc_decoder import CTCDecoder


class FuzzyCTCDecoder(CTCDecoder):
    def __init__(self):
        super().__init__()

    def decode(self, logits):
        # TODO: todo
        return None
