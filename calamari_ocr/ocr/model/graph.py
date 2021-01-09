import tensorflow as tf
from typing import List, Tuple


from calamari_ocr.ocr.model.params import ModelParams, LayerType, LayerParams
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import ctc_ops as ctc

from tfaip.base.model.graphbase import GraphBase

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


def pad(input_tensors):
    input, padding = input_tensors[0], input_tensors[1]
    px, py = padding
    shape = K.shape(input)
    static_shape = input.shape
    output = tf.image.pad_to_bounding_box(input, 0, 0,
                                          static_shape[1] or K.gather(shape, 1) + px,
                                          static_shape[2] or K.gather(shape, 2) + py)
    return output


class Graph(GraphBase):
    @classmethod
    def params_cls(cls):
        return ModelParams

    def __init__(self, params: ModelParams, name='CalamariGraph', **kwargs):
        super(Graph, self).__init__(params, name=name, **kwargs)

        self.conv_layers: List[Tuple[LayerParams, tf.keras.layers.Layer]] = []
        self.lstm_layers: List[Tuple[LayerParams, tf.keras.layers.Layer]] = []
        cnn_idx = 0
        for layer_index, layer in enumerate([l for l in params.layers if l.type != LayerType.LSTM]):
            if layer.type == LayerType.Convolutional:
                self.conv_layers.append((layer, KL.Conv2D(
                    name="conv2d_{}".format(cnn_idx),
                    filters=layer.filters,
                    kernel_size=(layer.kernel_size.x, layer.kernel_size.y),
                    padding="same",
                    activation="relu",
                )))
                cnn_idx += 1
            elif layer.type == LayerType.Concat:
                self.conv_layers.append((layer, KL.Concatenate(axis=-1)))
            elif layer.type == LayerType.DilatedBlock:
                depth = max(1, layer.dilated_depth)
                dilated_layers = [
                    KL.Conv2D(
                        name='conv2d_dilated{}_{}'.format(i, cnn_idx),
                        filters=layer.filters // depth,
                        kernel_size=(layer.kernel_size.x, layer.kernel_size.y),
                        padding="same",
                        activation="relu",
                        dilation_rate=2 ** (i + 1),
                    )
                    for i in range(depth)
                ]
                concat_layer = KL.Concatenate(axis=-1)
                cnn_idx += 1
                self.conv_layers.append((layer, (dilated_layers, concat_layer)))
            elif layer.type == LayerType.TransposedConv:
                self.conv_layers.append((layer, KL.Conv2DTranspose(
                    name="tconv2d_{}".format(cnn_idx),
                    filters=layer.filters,
                    kernel_size=(layer.kernel_size.x, layer.kernel_size.y),
                    strides=(layer.stride.x, layer.stride.y),
                    padding="same",
                    activation="relu",
                )))
                cnn_idx += 1
            elif layer.type == LayerType.MaxPooling:
                self.conv_layers.append((layer, KL.MaxPool2D(
                    name="pool2d_{}".format(layer_index),
                    pool_size=(layer.kernel_size.x, layer.kernel_size.y),
                    strides=(layer.stride.x, layer.stride.y),
                    padding="same",
                )))
            else:
                raise Exception("Unknown layer of type %s" % layer.type)

        for layer_index, layer in enumerate([l for l in params.layers if l.type == LayerType.LSTM]):
            lstm = KL.LSTM(
                units=layer.hidden_nodes,
                activation='tanh',
                recurrent_activation='sigmoid',
                recurrent_dropout=0,
                unroll=False,
                use_bias=True,
                return_sequences=True,
                unit_forget_bias=True,
                name=f'lstm_{layer_index}' if layer_index > 0 else 'lstm',
            )
            self.lstm_layers.append((layer, KL.Bidirectional(
                lstm,
                name='bidirectional',
                merge_mode='concat',
            )))

        self.dropout = KL.Dropout(params.dropout)
        self.logits = KL.Dense(params.classes, name='logits')
        self.softmax = KL.Softmax(name='softmax')

    def call(self, inputs, **kwargs):
        params: ModelParams = self._params
        input_data = tf.cast(inputs['img'], tf.float32) / 255.0
        input_sequence_length = K.flatten(inputs['img_len'])
        shape = input_sequence_length, -1

        # if concat or conv_T layers are present, we need to pad the input to ensure that possible upsampling layers work properly
        has_concat = any([l.type == LayerType.Concat or l.type == LayerType.TransposedConv for l in params.layers])
        if has_concat:
            sx, sy = 1, 1
            for layer_index, layer in enumerate(
                    [l for l in params.layers if l.type == LayerType.MaxPooling]):
                sx *= layer.stride.x
                sy *= layer.stride.y
            padding = calculate_padding(input_data, (sx, sy))
            padded = KL.Lambda(pad, name='padded_input')([input_data, padding])
            last_layer_output = padded
        else:
            last_layer_output = input_data

        layers_by_index = []
        for (lp, layer) in self.conv_layers:
            layers_by_index.append(last_layer_output)
            if lp.type == LayerType.Convolutional:
                last_layer_output = layer(last_layer_output)
            elif lp.type == LayerType.Concat:
                last_layer_output = layer([layers_by_index[i] for i in lp.concat_indices])
            elif lp.type == LayerType.DilatedBlock:
                ds = K.shape(last_layer_output)
                ss = last_layer_output.shape
                dilated_layers, concat_layer = layer
                dilated_layers = [dl(last_layer_output) for dl in dilated_layers]
                last_layer_output = concat_layer(dilated_layers)
                last_layer_output = K.reshape(last_layer_output, [ds[0], ds[1], ss[2], ss[3]])
            elif lp.type == LayerType.TransposedConv:
                last_layer_output = layer(last_layer_output)
            elif lp.type == LayerType.MaxPooling:
                last_layer_output = layer(last_layer_output)
                shape = (shape[0] // lp.stride.x, shape[1] // lp.stride.y)
            else:
                raise Exception("Unknown layer of type %s" % lp.type)

        lstm_seq_len, lstm_num_features = shape
        lstm_seq_len = K.cast(lstm_seq_len, 'int32')
        ds = K.shape(last_layer_output)
        ss = last_layer_output.shape
        last_layer_output = K.reshape(last_layer_output, (ds[0], ds[1], ss[2] * ss[3]))

        if len(self.lstm_layers) > 0:
            for lstm_params, lstm_layer in self.lstm_layers:
                last_layer_output = lstm_layer(last_layer_output)

        if params.dropout > 0:
            last_layer_output = self.dropout(last_layer_output)

        blank_last_logits = self.logits(last_layer_output)
        blank_last_softmax = self.softmax(blank_last_logits)

        logits = tf.roll(blank_last_logits, shift=1, axis=-1)
        softmax = tf.nn.softmax(logits)

        greedy_decoded = ctc.ctc_greedy_decoder(inputs=array_ops.transpose(blank_last_logits, perm=[1, 0, 2]),
                                                sequence_length=tf.cast(K.flatten(lstm_seq_len),
                                                                        'int32'))[0][0]

        return {
            'blank_last_logits': blank_last_logits,
            'blank_last_softmax': blank_last_softmax,
            'out_len': lstm_seq_len,
            'logits': logits,
            'softmax': softmax,
            'decoded': tf.sparse.to_dense(greedy_decoded, default_value=-1) + 1
        }

