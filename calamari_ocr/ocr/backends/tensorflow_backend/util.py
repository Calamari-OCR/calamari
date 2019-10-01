import numpy as np


def sparse_to_lists(sparse, shift_values=1):
    indices, values, dense_shape = sparse.indices, sparse.values, sparse.dense_shape
    return convert_targets_to_lists(indices, values, dense_shape, shift_values=shift_values)


def convert_targets_to_lists(indices, values, dense_shape, shift_values=1):
    out = [[] for _ in range(dense_shape[0])]

    for index, value in zip(indices, values):
        x, y = tuple(index)
        assert (len(out[x]) == y)  # consistency check
        out[x].append(value + shift_values)

    return [np.asarray(o, dtype=np.int64) for o in out]

