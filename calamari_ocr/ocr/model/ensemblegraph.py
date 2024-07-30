import logging

from tfaip.model.graphbase import GenericGraphBase

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops import ctc_ops as ctc

from calamari_ocr.ocr.model.graph import CalamariGraph
from calamari_ocr.ocr.model.params import ModelParams


logger = logging.getLogger(__name__)


class EnsembleGraph(GenericGraphBase[ModelParams]):
    def __init__(self, params: ModelParams, name="CalamariGraph", **kwargs):
        super(EnsembleGraph, self).__init__(params, name=name, **kwargs)

        self.fold_graphs = [CalamariGraph(params, f"voter_{i}") for i in range(params.ensemble)]
        if self._params.masking_mode > 0:
            logger.warning("Changed masking during training. This should only be used for evaluation!")

    def make_outputs(self, blank_last_softmax, lstm_seq_len, complete_outputs):
        softmax = tf.roll(blank_last_softmax, shift=1, axis=-1)

        greedy_decoded = ctc.ctc_greedy_decoder(
            inputs=tf.transpose(blank_last_softmax, perm=[1, 0, 2]),
            sequence_length=tf.cast(K.flatten(lstm_seq_len), "int32"),
        )[0][0]

        outputs = {
            "blank_last_logits": tf.math.log(blank_last_softmax),
            "blank_last_softmax": blank_last_softmax,
            "logits": tf.math.log(softmax),
            "softmax": softmax,
            "out_len": lstm_seq_len,
            "decoded": tf.sparse.to_dense(greedy_decoded, default_value=-1) + 1,
        }

        for i, voter_output in enumerate(complete_outputs):
            for k, v in voter_output.items():
                outputs[f"{k}_{i}"] = v

        return outputs

    def build_prediction_graph(self, inputs, training=None):
        # Prediction Graph: standard voting
        complete_outputs = [self.fold_graphs[i].predict(inputs) for i in range(len(self.fold_graphs))]

        lstm_seq_len = complete_outputs[0]["out_len"]  # is the same for all children
        softmax_outputs = tf.stack([out["blank_last_softmax"] for out in complete_outputs], axis=0)

        blank_last_softmax = tf.reduce_mean(softmax_outputs, axis=0)
        return self.make_outputs(blank_last_softmax, lstm_seq_len, complete_outputs)

    def build_train_graph(self, inputs, targets, training=None):
        if training is None:
            training = K.learning_phase()

        batch_size = tf.shape(inputs["img_len"])[0]
        max_lstm_seq_len = self._params.compute_downscaled((tf.shape(inputs["img"])[1], 1))[0]

        # Training/Validation graph
        def training_step():
            tf.debugging.assert_greater_equal(targets["fold_id"], 0)
            complete_outputs = [self.fold_graphs[i].train(inputs, targets) for i in range(len(self.fold_graphs))]

            lstm_seq_len = complete_outputs[0]["out_len"]  # is the same for all children
            softmax_outputs = tf.stack([out["blank_last_softmax"] for out in complete_outputs], axis=0)

            # Training: Mask out network that does not contribute to a sample to generate strong voters
            if self._params.masking_mode == 0:
                # Fixed fold ID
                mask = [tf.not_equal(i, targets["fold_id"]) for i in range(len(self.fold_graphs))]
                softmax_outputs *= tf.cast(tf.expand_dims(mask, axis=-1), dtype="float32")
                blank_last_softmax = tf.reduce_sum(softmax_outputs, axis=0) / (
                    len(self.fold_graphs) - 1
                )  # only n - 1 since one voter is 0
            elif self._params.masking_mode == 1:
                # No fold ID
                # In this case, training behaves similar to prediction
                blank_last_softmax = tf.reduce_mean(softmax_outputs, axis=0)
            elif self._params.masking_mode == 2:
                # Random fold ID
                fold_id = tf.random.uniform(
                    minval=0,
                    maxval=len(self.fold_graphs),
                    dtype="int32",
                    shape=[batch_size, 1],
                )
                mask = [tf.not_equal(i, fold_id) for i in range(len(self.fold_graphs))]
                softmax_outputs *= tf.cast(tf.expand_dims(mask, axis=-1), dtype="float32")
                blank_last_softmax = tf.reduce_sum(softmax_outputs, axis=0) / (
                    len(self.fold_graphs) - 1
                )  # only n - 1 since one voter is 0
            else:
                raise NotImplementedError
            return blank_last_softmax, lstm_seq_len, complete_outputs

        def validation_step():
            # any dummy output is max length, to get actional outpu length t use reduce_min
            def gen_empty_output(bs):
                empty = tf.zeros(shape=[bs, max_lstm_seq_len, self._params.classes], dtype="float32")
                return {
                    "blank_last_logits": empty,
                    "blank_last_softmax": empty,
                    "out_len": tf.repeat(max_lstm_seq_len, repeats=bs),
                    "logits": empty,
                    "softmax": empty,
                    "decoded": tf.zeros(shape=[bs, max_lstm_seq_len], dtype="int64"),
                }

            empty_output = gen_empty_output(1)

            # Validation: Compute output for each graph but only for its own partition
            # Per sample this is one CER which is then used e. g. for early stopping
            def apply_single_model(batch):
                batch = batch["out_len"]  # Take any, all are batch id as input
                single_batch_data = {k: [tf.gather(v, batch)] for k, v in inputs.items()}
                complete_outputs = [
                    tf.cond(
                        tf.equal(i, targets["fold_id"][batch]),
                        lambda: self.fold_graphs[i].train(single_batch_data, None),
                        lambda: empty_output,
                    )
                    for i in range(len(self.fold_graphs))
                ]
                outputs = {
                    k: tf.gather(
                        tf.stack([out[k] for out in complete_outputs]),
                        targets["fold_id"][batch][0],
                    )[0]
                    for k in empty_output.keys()
                    if k != "decoded"
                }
                paddings = [([0, 0], [0, max_lstm_seq_len - tf.shape(out["decoded"])[1]]) for out in complete_outputs]
                outputs["decoded"] = tf.gather(
                    tf.stack(
                        [
                            tf.pad(out["decoded"], padding, "CONSTANT", constant_values=0)
                            for out, padding in zip(complete_outputs, paddings)
                        ]
                    ),
                    targets["fold_id"][batch][0],
                )[0]
                return outputs

            complete_outputs = tf.map_fn(
                apply_single_model,
                {k: tf.range(batch_size, dtype=v.dtype) for k, v in empty_output.items()},
                parallel_iterations=len(self.fold_graphs),
                back_prop=False,
            )
            return (
                complete_outputs["blank_last_softmax"],
                complete_outputs["out_len"],
                [complete_outputs] * len(self.fold_graphs),
            )

        if isinstance(training, bool) or isinstance(training, int):
            blank_last_softmax, lstm_seq_len, complete_outputs = training_step() if training else validation_step()
        else:
            blank_last_softmax, lstm_seq_len, complete_outputs = tf.cond(training, training_step, validation_step)

        return self.make_outputs(blank_last_softmax, lstm_seq_len, complete_outputs)
