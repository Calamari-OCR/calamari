import tensorflow as tf
import numpy as np
import json
from typing import Generator


from calamari_ocr.ocr.backends.model_interface import ModelInterface, NetworkPredictionResult
from calamari_ocr.ocr.callbacks import TrainingCallback
from calamari_ocr.proto import LayerParams, NetworkParams
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import ctc_ops as ctc
from .callbacks.visualize import VisCallback
from .callbacks.earlystopping import EarlyStoppingCallback

keras = tf.keras
K = keras.backend
KL = keras.layers
Model = keras.Model


class TensorflowModel(ModelInterface):
    def __init__(self, network_proto, graph_type="train", ctc_decoder_params=None, batch_size=1,
                 codec=None, processes=1):
        super().__init__(network_proto, graph_type, ctc_decoder_params, batch_size,
                         codec=codec, processes=processes)
        self.downscale_factor = 1  # downscaling factor of the inputs due to pooling layers
        self.input_data = KL.Input(name='input_data', shape=(None, network_proto.features, self.input_channels))
        self.input_length = KL.Input(name='input_sequence_length', shape=(1,))
        self.input_params = KL.Input(name='input_data_params', shape=(1,), dtype='string')
        self.targets = KL.Input(name='targets', shape=[None], dtype='int32')
        self.targets_length = KL.Input(name='targets_length', shape=[1], dtype='int64')
        self.output_seq_len, self.logits, self.softmax, self.scale_factor, self.sparse_decoded = \
            self.create_network(self.network_proto.dropout, self.input_data, self.input_length)
        if graph_type == "train":
            self.model = self.create_solver()
        else:
            self.model = self.create_predictor()

    def create_predictor(self):
        return Model(inputs=[
            self.input_data, self.input_length, self.input_params
        ], outputs=[
            self.softmax, self.input_params, self.output_seq_len])

    def create_network(self, dropout_rate, input_data, input_sequence_length):
        network_proto = self.network_proto
        factor = 1
        shape = input_sequence_length, network_proto.features

        last_num_filters = 1

        last_layer = input_data
        cnn_idx = 0
        for layer_index, layer in enumerate([l for l in network_proto.layers if l.type != LayerParams.LSTM]):
            if layer.type == LayerParams.CONVOLUTIONAL:
                last_layer = KL.Conv2D(
                    name="conv2d_{}".format(cnn_idx),
                    filters=layer.filters,
                    kernel_size=(layer.kernel_size.x, layer.kernel_size.y),
                    padding="same",
                    activation="relu",
                )(last_layer)
                last_num_filters = layer.filters
                cnn_idx += 1
            elif layer.type == LayerParams.MAX_POOLING:
                last_layer = KL.MaxPool2D(
                    name="pool2d_{}".format(layer_index),
                    pool_size=(layer.kernel_size.x, layer.kernel_size.y),
                    strides=(layer.stride.x, layer.stride.y),
                    padding="same",
                )(last_layer)
                shape = (shape[0] // layer.stride.x, shape[1] // layer.stride.y)
                factor *= layer.stride.x
            else:
                raise Exception("Unknown layer of type %s" % layer.type)

        self.downscale_factor = factor
        lstm_seq_len, lstm_num_features = shape
        lstm_seq_len = K.cast(lstm_seq_len, 'int32')
        last_layer = KL.Reshape((-1, last_num_filters * lstm_num_features))(last_layer)

        # lstm_num_features = last_num_filters * lstm_num_features

        lstm_layers = [l for l in network_proto.layers if l.type == LayerParams.LSTM]

        if len(lstm_layers) > 0:
            for i, lstm in enumerate(lstm_layers):
                if lstm.hidden_nodes != lstm_layers[0].hidden_nodes:
                    raise Exception("Currently all lstm layers must have an equal number of hidden nodes. "
                                    "Got {} != {}".format(lstm.hidden_nodes, lstm_layers[0].hidden_nodes))

                last_layer = KL.Bidirectional(KL.LSTM(
                    units=lstm_layers[0].hidden_nodes,
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    recurrent_dropout=0,
                    unroll=False,
                    use_bias=True,
                    return_sequences=True,
                    unit_forget_bias=True,
                ),
                    merge_mode='concat',
                )(last_layer)

        if network_proto.dropout > 0:
            last_layer = KL.Dropout(dropout_rate)(last_layer)

        logits = KL.Dense(network_proto.classes, name='logits')(last_layer)
        softmax = KL.Softmax(name='softmax')(logits)

        def sparse_decoded(logits, output_seq_len):
            return ctc.ctc_greedy_decoder(inputs=array_ops.transpose(logits, perm=[1, 0, 2]),
                                          sequence_length=tf.cast(K.flatten(output_seq_len),
                                                                  'int32'))[0][0]

        sparse_decoded = KL.Lambda(lambda args: sparse_decoded(*args), name='sparse_decoded')(
            (logits, lstm_seq_len))

        return lstm_seq_len, logits, softmax, factor, sparse_decoded

    def create_dataset_inputs(self, input_dataset, batch_size, line_height, max_buffer_size=1000, mode=None):
        if not mode:
            mode = self.graph_type

        buffer_size = len(input_dataset) if input_dataset else 10
        buffer_size = min(max_buffer_size, buffer_size) if max_buffer_size > 0 else buffer_size
        input_channels = self.input_channels

        def gen():
            epochs = 1
            for i, l, d in input_dataset.generator(epochs):
                if i is None:
                    continue

                if mode == "train" and len(l) == 0:
                    continue

                l = np.array(self.codec.encode(l) if l else np.zeros((0, ), dtype='int32'))

                # gray or binary input, add missing axis
                if len(i.shape) == 2:
                    i = np.expand_dims(i, axis=-1)

                if i.shape[-1] != input_channels:
                    raise ValueError("Expected {} channels but got {}. Shape of input {}".format(
                        input_channels, i.shape[-1], i.shape))

                # tensorflow ctc loss expects last label as blank
                l -= 1

                if mode == 'train' and len(i) // self.downscale_factor < len(l):
                    # skip longer outputs than inputs
                    continue

                yield i / 255.0, l, [len(i)], [len(l)], [json.dumps(d)]

        dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32, tf.int32, tf.int32, tf.string))
        if mode == "train":
            dataset = dataset.repeat().shuffle(buffer_size, seed=self.network_proto.backend.random_seed)
        else:
            pass

        dataset = dataset.padded_batch(batch_size, ([None, line_height, input_channels], [None], [1], [1], [1]),
                                       padding_values=(np.float32(0), np.int32(-1), np.int32(0), np.int32(0), ''))

        def group(data, targets, len_data, len_labels, user_data):
            return (
                {"input_data": data, "input_sequence_length": len_data, "input_data_params": user_data, "targets": targets, "targets_length": len_labels},
                {'ctc': np.zeros([batch_size]), 'cer_acc': np.zeros([batch_size])}
            )

        return dataset.prefetch(5).map(group)

    def create_solver(self):
        def sparse_targets(targets, targets_length):
            return tf.cast(K.ctc_label_dense_to_sparse(targets, math_ops.cast(
                K.flatten(targets_length), dtype='int32')), 'int32')

        def create_cer(sparse_decoded, sparse_targets):
            return tf.edit_distance(tf.cast(sparse_decoded, tf.int32), sparse_targets, normalize=True)

        # Note for codec change: the codec size is derived upon creation, therefore the ctc ops must be created
        # using the true codec size (the W/B-Matrix may change its shape however during loading/codec change
        # to match the true codec size
        loss = KL.Lambda(lambda args: K.ctc_batch_cost(*args), output_shape=(1,), name='ctc')((self.targets, self.softmax, self.output_seq_len, self.targets_length))
        self.sparse_targets = KL.Lambda(lambda args: sparse_targets(*args), name='sparse_targets')((self.targets, self.targets_length))
        self.cer = KL.Lambda(lambda args: create_cer(*args), output_shape=(1,), name='cer')((self.sparse_decoded, self.sparse_targets))

        if self.network_proto.solver == NetworkParams.MOMENTUM_SOLVER:
            optimizer = keras.optimizers.SGD(self.network_proto.learning_rate, self.network_proto.momentum, clipnorm=self.network_proto.clipping_norm)
        elif self.network_proto.solver == NetworkParams.ADAM_SOLVER:
            optimizer = keras.optimizers.Adam(self.network_proto.learning_rate, clipnorm=self.network_proto.clipping_norm)
        else:
            raise Exception("Unknown solver of type '%s'" % self.network_proto.solver)

        def ctc_loss(t, p):
            return p

        model = Model(inputs=[self.targets, self.input_data, self.input_length, self.targets_length], outputs=[loss])
        model.compile(optimizer=optimizer, loss={'ctc': ctc_loss},
                      )

        return model

    def load_weights(self, filepath):
        self.model.load_weights(filepath + '.h5')

    def copy_weights_from_model(self, model, indices_to_delete, indices_to_add):
        for target_layer, source_layer in zip(self.model.layers, model.model.layers):
            target_weights = target_layer.weights
            source_weights = source_layer.weights
            if len(target_weights) != len(source_weights):
                raise Exception("Different network structure detected.")

            if len(target_weights) == 0:
                continue

            if target_layer.name.startswith('logits'):
                tW, sW = [(tw, sw) for tw, sw in zip(target_weights, source_weights) if 'kernel' in tw.name][0]
                tB, sB = [(tw, sw) for tw, sw in zip(target_weights, source_weights) if 'bias' in tw.name][0]

                W_val = np.delete(sW.value(), [i - 1 for i in indices_to_delete], axis=1)
                B_val = np.delete(sB.value(), [i - 1 for i in indices_to_delete], axis=0)

                # add new indices at the end
                if list(range(W_val.shape[1], W_val.shape[1] + len(indices_to_add))) != list(sorted(indices_to_add)):
                    raise Exception("Additional labels must be added at the end, but got label indices {} != {}".format(
                        range(W_val.shape[1], W_val.shape[1] + len(indices_to_add)), sorted(indices_to_add)))

                W_val = np.concatenate(
                    (W_val[:, :-1], np.random.uniform(-0.1, 0.1, (W_val.shape[0], len(indices_to_add))), W_val[:, -1:]),
                    axis=1)
                B_val = np.concatenate((B_val[:-1], np.zeros((len(indices_to_add),)), B_val[-1:]), axis=0)

                # reassign values
                tW.assign(W_val)
                tB.assign(B_val)
            else:
                for tw, sw in zip(target_weights, source_weights):
                    tw.assign(sw)

    def train(self, dataset, validation_dataset, checkpoint_params, text_post_proc, progress_bar,
              training_callback=TrainingCallback()):
        dataset_gen = self.create_dataset_inputs(dataset, self.batch_size, self.network_proto.features, self.network_proto.backend.shuffle_buffer_size)
        if validation_dataset:
            val_dataset_gen = self.create_dataset_inputs(validation_dataset, self.batch_size, self.network_proto.features, self.network_proto.backend.shuffle_buffer_size, mode='test')
        else:
            val_dataset_gen = None

        predict_func = K.function({t.op.name: t for t in [self.input_data, self.input_length, self.input_params, self.targets, self.targets_length]}, [self.cer, self.sparse_targets, self.sparse_decoded])
        steps_per_epoch = max(1, int(dataset.epoch_size() / checkpoint_params.batch_size))
        v_cb = VisCallback(training_callback, self.codec, dataset_gen, predict_func, checkpoint_params, steps_per_epoch, text_post_proc)
        es_cb = EarlyStoppingCallback(training_callback, self.codec, val_dataset_gen, predict_func, checkpoint_params, 0 if not validation_dataset else max(1, int(np.ceil(validation_dataset.epoch_size() / checkpoint_params.batch_size))), v_cb, progress_bar)

        self.model.fit(
            dataset_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=1000,
            use_multiprocessing=False,
            shuffle=False,
            verbose=0,
            callbacks=[
                v_cb, es_cb
            ]
        )

    def predict_raw_batch(self, x: np.array, len_x: np.array) -> Generator[NetworkPredictionResult, None, None]:
        out = self.model.predict_on_batch(
            [x / 255, len_x, np.zeros((len(x), 1), dtype=np.str)],
        )
        for sm, params, sl in zip(*out):
            sl = sl[0]
            sm = np.roll(sm, 1, axis=1)
            decoded = self.ctc_decoder.decode(sm[:sl])
            pred = NetworkPredictionResult(softmax=sm,
                                           output_length=sl,
                                           decoded=decoded,
                                           )
            yield pred

    def predict_dataset(self, dataset) -> Generator[NetworkPredictionResult, None, None]:
        dataset_gen = self.create_dataset_inputs(dataset, self.batch_size, self.network_proto.features, self.network_proto.backend.shuffle_buffer_size,
                                                 mode='test')
        out = self.model.predict(
            dataset_gen,
        )
        for softmax, params, output_seq_len in zip(*out):
            softmax = np.roll(softmax, 1, axis=1)  # fix bla
            # decode encoded params from json. On python<=3.5 this are bytes, else it already is a str
            enc_param = params[0]
            enc_param = json.loads(enc_param.decode("utf-8") if isinstance(enc_param, bytes) else enc_param)
            decoded = self.ctc_decoder.decode(softmax[:output_seq_len[0]])
            # return prediction result
            yield NetworkPredictionResult(softmax=softmax,
                                          output_length=output_seq_len,
                                          decoded=decoded,
                                          params=enc_param,
                                          ground_truth=None,
                                          )

    def output_to_input_position(self, x):
        return x * self.scale_factor
