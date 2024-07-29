from functools import partial

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import ctc_ops as ctc
from tfaip.model.graphbase import GraphBase

from calamari_ocr.ocr.model.layers.concat import ConcatLayerParams
from calamari_ocr.ocr.model.layers.toinputdims import ToInputDimsLayerParams
from calamari_ocr.ocr.model.layers.transposedconv2d import TransposedConv2DLayerParams
from calamari_ocr.ocr.model.params import ModelParams

keras = tf.keras
K = keras.backend
KL = keras.layers


def calculate_padding(input, scaling_factor):
    def scale(i, f):
        return (f - i % f) % f

    shape = input.shape
    dyn_shape = K.shape(input)
    px = scale(shape[1] or K.gather(dyn_shape, 1), scaling_factor[0])
    py = scale(shape[2] or K.gather(dyn_shape, 2), scaling_factor[1])
    return px, py


def pad(input_tensors, x_only=False):
    input, padding = input_tensors[0], input_tensors[1]
    px, py = padding
    shape = K.shape(input)
    static_shape = input.shape
    if x_only:
        output = tf.image.pad_to_bounding_box(
            input, 0, 0, (static_shape[1] or K.gather(shape, 1)) + px, static_shape[2]
        )
    else:
        output = tf.image.pad_to_bounding_box(
            input,
            0,
            0,
            (static_shape[1] or K.gather(shape, 1)) + px,
            (static_shape[2] or K.gather(shape, 2)) + py,
        )
    return output


class CalamariGraph(GraphBase[ModelParams]):
    def __init__(self, params: ModelParams, name="CalamariGraph", **kwargs):
        super().__init__(params, name=name, **kwargs)
        assert params.classes > 0, "non initialized number of classes"

        self.layer_instances = [l.create() for l in params.layers]

        self.reshape = ToInputDimsLayerParams(dims=3).create()
        self.logits = KL.Dense(params.classes, name="logits")
        self.softmax = KL.Softmax(name="softmax")
        self.temperature = (
            tf.constant(params.temperature, dtype=tf.float32, name="temperature") if params.temperature > 0 else None
        )

    def build_graph(self, inputs, training=None):
        params: ModelParams = self._params
        input_data = tf.cast(inputs["img"], tf.float32) / 255.0
        input_sequence_length = K.flatten(inputs["img_len"])
        shape = input_sequence_length, -1

        # if concat or conv_T layers are present, we need to pad the input to ensure that possible
        # up-sampling layers work properly
        require_padding = any([isinstance(l, (ConcatLayerParams, TransposedConv2DLayerParams)) for l in params.layers])
        if require_padding:
            s = self._params.compute_max_downscale_factor()
            padding = calculate_padding(input_data, s.to_tuple())
            padded = KL.Lambda(partial(pad, x_only=True), name="padded_input")([input_data, padding])
            last_layer_output = padded
        else:
            last_layer_output = input_data

        layers_outputs_by_index = []
        for layer in self.layer_instances:
            layers_outputs_by_index.append(last_layer_output)
            if isinstance(layer.params, ConcatLayerParams):
                last_layer_output = layer(layers_outputs_by_index)
            else:
                last_layer_output = layer(last_layer_output)

        lstm_seq_len, lstm_num_features = self._params.compute_downscaled(shape)
        lstm_seq_len = K.cast(lstm_seq_len, "int32")

        last_layer_output = self.reshape(last_layer_output)
        blank_last_logits = self.logits(last_layer_output)
        blank_last_softmax = self.softmax(blank_last_logits)

        logits = tf.roll(blank_last_logits, shift=1, axis=-1)
        if self.temperature != None:
            logits = tf.divide(logits, self.temperature)  ### TEST scale, seems to work...

        softmax = tf.nn.softmax(logits)

        greedy_decoded = ctc.ctc_greedy_decoder(
            inputs=array_ops.transpose(blank_last_logits, perm=[1, 0, 2]),
            sequence_length=tf.cast(K.flatten(lstm_seq_len), "int32"),
        )[0][0]

        return {
            "blank_last_logits": blank_last_logits,
            "blank_last_softmax": blank_last_softmax,
            "out_len": lstm_seq_len,
            "logits": logits,
            "softmax": softmax,
            "decoded": tf.sparse.to_dense(greedy_decoded, default_value=-1) + 1,
        }

    @classmethod
    def from_config(cls, config):
        try:
            return super().from_config(config)
        except TypeError:
            # convert old format?
            from calamari_ocr.ocr.savedmodel.migrations.version3to4 import (
                migrate_model_params,
            )

            migrate_model_params(config["params"])
            return super().from_config(config)
