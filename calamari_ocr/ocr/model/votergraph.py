from tfaip.base.model.graphbase import GraphBase

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops import ctc_ops as ctc

from calamari_ocr.ocr.model.graph import Graph
from calamari_ocr.ocr.model.params import ModelParams


class VoterGraph(GraphBase):
    @classmethod
    def params_cls(cls):
        return ModelParams

    def __init__(self, params: ModelParams, name='CalamariGraph', **kwargs):
        super(VoterGraph, self).__init__(params, name=name, **kwargs)
        n_folds = 5
        self.fold_graphs = [Graph(params, f"voter_{i}") for i in range(n_folds)]

    def call(self, inputs, training=None, **kwargs):
        # only pass folds to selected folds
        complete_outputs = [self.fold_graphs[i](inputs) for i in range(len(self.fold_graphs))]

        lstm_seq_len = complete_outputs[0]['out_len']   # is the same for all children
        softmax_outputs = tf.stack([out['blank_last_softmax'] for out in complete_outputs], axis=0)

        if training:
            # Training: Mask out network that does not contribute to a sample to generate strong voters
            mask = [tf.not_equal(i, inputs['fold_id']) for i in range(len(self.fold_graphs))]
            softmax_outputs *= tf.cast(tf.expand_dims(mask, axis=-1), dtype='float32')
            blank_last_softmax = tf.reduce_sum(softmax_outputs, axis=0) / 4.0  # only 4 since one voter is 0
        else:
            # standard voting
            blank_last_softmax = tf.reduce_mean(softmax_outputs, axis=0)

        softmax = tf.roll(blank_last_softmax, shift=1, axis=-1)

        greedy_decoded = ctc.ctc_greedy_decoder(inputs=tf.transpose(blank_last_softmax, perm=[1, 0, 2]),
                                                sequence_length=tf.cast(K.flatten(lstm_seq_len),
                                                                        'int32'))[0][0]

        return {
            'blank_last_logits': tf.math.log(blank_last_softmax),
            'blank_last_softmax': blank_last_softmax,
            'logits': tf.math.log(softmax),
            'softmax': softmax,
            "out_len": lstm_seq_len,
            'decoded': tf.sparse.to_dense(greedy_decoded, default_value=-1) + 1,
        }
