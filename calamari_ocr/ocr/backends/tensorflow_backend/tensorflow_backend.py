import numpy as np

from calamari_ocr.ocr.backends.backend_interface import BackendInterface
from calamari_ocr.ocr.backends.tensorflow_backend.tensorflow_model import TensorflowModel


class TensorflowBackend(BackendInterface):
    def __init__(self,
                 network_proto,
                 restore,
                 weights):
        super().__init__(network_proto)
        if restore:
            self._model = TensorflowModel.load(network_proto, restore)
        else:
            self._model = TensorflowModel.from_proto(network_proto)

        if weights:
            self._model.load_weights(weights)

    def prepare(self, train):
        self._model.prepare(train)

    def train(self, batch_x, batch_y):
        x, len_x = TensorflowBackend.__sparse_data_to_dense(batch_x)
        y = TensorflowBackend.__to_sparse_matrix(batch_y)

        cost, optimizer, logits, seq_len, ler, decoded = self._model.train(x, len_x, y)
        logits = np.roll(logits, 1, axis=2)
        return {
            "loss": cost,
            "logits": logits,
            "ler": ler,
            "decoded": TensorflowBackend.__sparse_to_lists(decoded),
            "gt": batch_y,
            "logits_lengths": seq_len,
        }

    def predict(self, batch_x):
        x, len_x = TensorflowBackend.__sparse_data_to_dense(batch_x)
        logits, seq_len, decoded = self._model.predict(x, len_x)
        logits = np.roll(logits, 1, axis=2)
        # decoded = TensorflowBackend.__sparse_to_lists(decoded)
        return [self.ctc_decoder.decode(l[:s]) for l, s in zip(logits, seq_len)]

    def save_checkpoint(self, filepath):
        self._model.save(filepath)

    def load_checkpoint_weights(self, filepath):
        self._model.load_weights(filepath)

    def realign_model_labels(self, indices_to_delete, indices_to_add):
        self._model.realign_labels(indices_to_delete, indices_to_add)

    @staticmethod
    def __to_sparse_matrix(y, shift_values=-1):
        batch_size = len(y)
        indices = np.concatenate([np.concatenate(
            [
                np.full((len(y[i]), 1), i),
                np.reshape(range(len(y[i])), (-1, 1))
            ], 1) for i in range(batch_size)], 0)
        values = np.concatenate(y, 0) + shift_values
        dense_shape = np.asarray([batch_size, max([len(yi) for yi in y])])
        assert(len(indices) == len(values))

        return indices, values, dense_shape

    @staticmethod
    def __sparse_data_to_dense(x):
        batch_size = len(x)
        len_x = [xb.shape[0] for xb in x]
        max_line_length = max(len_x)

        # transform into batch (batch size, T, height)
        full_x = np.zeros((batch_size, max_line_length, x[0].shape[1]))
        for batch, xb in enumerate(x):
            full_x[batch, :len(xb)] = xb

        # return full_x, len_x
        return full_x, [l for l in len_x]

    @staticmethod
    def __sparse_to_lists(sparse, shift_values=1):
        indices, values, dense_shape = sparse

        out = [[] for _ in range(dense_shape[0])]

        for index, value in zip(indices, values):
            x, y = tuple(index)
            assert(len(out[x]) == y)  # consistency check
            out[x].append(value + shift_values)

        return out

