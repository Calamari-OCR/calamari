from tfaip.base.model.graphbase import GraphBase

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops import ctc_ops as ctc

from calamari_ocr.ocr.model.graph import Graph
from calamari_ocr.ocr.model.params import ModelParams


class Intermediate(tf.keras.layers.Layer):
    def __init__(self, params: ModelParams, name='CalamariGraph', **kwargs):
        super(Intermediate, self).__init__(name=name, **kwargs)
        self._params = params
        self.fold_graphs = [Graph(params, f"voter_{i}") for i in range(params.voters)]

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()
        batch_size = tf.shape(inputs['img_len'])[0]
        max_lstm_seq_len = self._params.compute_downscaled(tf.shape(inputs['img'])[1])
        # only pass folds to selected folds
        if 'fold_id' in inputs:
            def empty_output():
                # any dummy output is max length, to get actional outpu length t use reduce_min
                empty = tf.zeros(shape=[batch_size, max_lstm_seq_len, self._params.classes])

                return {
                    'blank_last_logits': empty,
                    'blank_last_softmax': empty,
                    'out_len': tf.repeat(max_lstm_seq_len, repeats=batch_size),
                    'logits': empty,
                    'softmax': empty,
                    'decoded': tf.zeros(shape=[batch_size, max_lstm_seq_len], dtype='int64'),
                }
            # Training/Validation graph
            def training_step():
                complete_outputs = [self.fold_graphs[i](inputs) for i in range(len(self.fold_graphs))]

                lstm_seq_len = complete_outputs[0]['out_len']  # is the same for all children
                softmax_outputs = tf.stack([out['blank_last_softmax'] for out in complete_outputs], axis=0)

                # Training: Mask out network that does not contribute to a sample to generate strong voters
                mask = [tf.not_equal(i, inputs['fold_id']) for i in range(len(self.fold_graphs))]
                softmax_outputs *= tf.cast(tf.expand_dims(mask, axis=-1), dtype='float32')
                blank_last_softmax = tf.reduce_sum(softmax_outputs, axis=0) / (len(self.fold_graphs) - 1)  # only n - 1 since one voter is 0
                return blank_last_softmax, lstm_seq_len, complete_outputs

            def validation_step():
                # Validation: Compute output for each graph but only for its own partition
                # Per sample this is one CER which is then used e. g. for early stopping
                complete_outputs = [tf.cond(tf.equal(i, inputs['fold_id'][0]), lambda: self.fold_graphs[i](inputs), empty_output) for i in range(len(self.fold_graphs))]
                seq_lens = [out['out_len'] for out in complete_outputs]
                lstm_seq_len = tf.reshape(tf.reduce_min(seq_lens, axis=0), shape=[batch_size])
                softmax_outputs = [out['blank_last_softmax'] for i, out in enumerate(complete_outputs)]
                blank_last_softmax = tf.gather(softmax_outputs, inputs['fold_id'][0])[0]
                return blank_last_softmax, lstm_seq_len, complete_outputs

            if isinstance(training, bool) or isinstance(training, int):
                blank_last_softmax, lstm_seq_len, complete_outputs = training_step() if training else validation_step()
            else:
                blank_last_softmax, lstm_seq_len, complete_outputs = tf.cond(training, training_step, validation_step)
        else:
            # Prediction Graph: standard voting
            complete_outputs = [self.fold_graphs[i](inputs) for i in range(len(self.fold_graphs))]

            lstm_seq_len = complete_outputs[0]['out_len']  # is the same for all children
            softmax_outputs = tf.stack([out['blank_last_softmax'] for out in complete_outputs], axis=0)

            blank_last_softmax = tf.reduce_mean(softmax_outputs, axis=0)

        softmax = tf.roll(blank_last_softmax, shift=1, axis=-1)

        greedy_decoded = ctc.ctc_greedy_decoder(inputs=tf.transpose(blank_last_softmax, perm=[1, 0, 2]),
                                                sequence_length=tf.cast(K.flatten(lstm_seq_len),
                                                                        'int32'))[0][0]

        outputs = {
            'blank_last_logits': tf.math.log(blank_last_softmax),
            'blank_last_softmax': blank_last_softmax,
            'logits': tf.math.log(softmax),
            'softmax': softmax,
            "out_len": lstm_seq_len,
            'decoded': tf.sparse.to_dense(greedy_decoded, default_value=-1) + 1,
        }

        for i, voter_output in enumerate(complete_outputs):
            for k, v in voter_output.items():
                outputs[f"{k}_{i}"] = v

        return outputs


class VoterGraph(GraphBase):
    @classmethod
    def params_cls(cls):
        return ModelParams

    def __init__(self, params: ModelParams, name='CalamariGraph', **kwargs):
        super(VoterGraph, self).__init__(params, name=name, **kwargs)
        self.wrapper = Intermediate(params)

    def call(self, inputs, **kwargs):
        return self.wrapper(inputs, **kwargs)
