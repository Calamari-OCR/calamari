class SequenceRecognizer:
    def __init__(self,
                 codec,
                 backend,
                 ):
        self.codec = codec
        self.backend = backend

    def train(self,
              batch_x,
              batch_y,
              ):
        result = self.backend.train_sequence(batch_x, batch_y)

        return result

