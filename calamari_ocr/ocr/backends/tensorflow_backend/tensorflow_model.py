import sys
import tensorflow as tf
import tensorflow.contrib.cudnn_rnn as cudnn_rnn
from tensorflow.python.ops import ctc_ops
import numpy as np

from calamari_ocr.ocr.backends.model_interface import ModelInterface
from calamari_ocr.proto import LayerParams, NetworkParams


class TensorflowModel(ModelInterface):
    def __init__(self, network_proto, graph, session, graph_type="train", batch_size=1, reuse_weights=False):
        super().__init__(network_proto, graph_type, batch_size, implementation_handles_batching=True)
        self.graph = graph
        self.session = session
        self.gpu_available = any([d.device_type == "GPU" for d in self.session.list_devices()])

        # load fuzzy ctc module if available
        if len(network_proto.backend.fuzzy_ctc_library_path) > 0 and network_proto.ctc == NetworkParams.CTC_FUZZY:
            from calamari_ocr.ocr.backends.tensorflow_backend.tensorflow_fuzzy_ctc_loader import load as load_fuzzy
            self.fuzzy_module = load_fuzzy(network_proto.backend.fuzzy_ctc_library_path)
        else:
            self.fuzzy_module = None

        # create graph
        with self.graph.as_default():
            tf.set_random_seed(self.network_proto.backend.random_seed)

            # inputs either as placeholders or as data set (faster)
            if self.implementation_handles_batching:
                self.inputs, self.input_seq_len, self.targets, self.dropout_rate, self.data_iterator = \
                    self.create_dataset_inputs(batch_size, network_proto.features)
            else:
                self.data_iterator = None
                self.inputs, self.input_seq_len, self.targets, self.dropout_rate = self.create_placeholders()

            # create network and solver (if train)
            if graph_type == "train":
                self.output_seq_len, self.time_major_logits, self.time_major_softmax, self.logits, self.softmax, self.decoded, self.sparse_decoded, self.scale_factor = \
                    self.create_network(self.inputs, self.input_seq_len, self.dropout_rate, reuse_variables=reuse_weights)
                self.train_op, self.loss, self.cer = self.create_solver(self.targets, self.time_major_logits, self.logits, self.output_seq_len, self.decoded)
            elif graph_type == "test":
                self.output_seq_len, self.time_major_logits, self.time_major_softmax, self.logits, self.softmax, self.decoded, self.sparse_decoded, self.scale_factor = \
                    self.create_network(self.inputs, self.input_seq_len, self.dropout_rate, reuse_variables=reuse_weights)
                self.cer = self.create_cer(self.decoded, self.targets)
            else:
                self.output_seq_len, self.time_major_logits, self.time_major_softmax, self.logits, self.softmax, self.decoded, self.sparse_decoded, self.scale_factor = \
                    self.create_network(self.inputs, self.input_seq_len, self.dropout_rate, reuse_variables=reuse_weights)

    def is_gpu_available(self):
        # create a dummy session and list available devices
        # search if a GPU is available
        gpu_enabled = False
        for d in self.session.list_devices():
            if d.device_type == "GPU":
                gpu_enabled = True
                break

        return gpu_enabled

    def create_network(self, inputs, input_seq_len, dropout_rate, reuse_variables):
        network_proto = self.network_proto
        seq_len = input_seq_len
        batch_size = tf.shape(inputs)[0]
        gpu_enabled = self.gpu_available

        with tf.variable_scope("", reuse=tf.AUTO_REUSE) as scope:
            no_layers = len(network_proto.layers) == 0
            if not no_layers:
                has_conv_or_pool = network_proto.layers[0].type != LayerParams.LSTM
            else:
                has_conv_or_pool = False

            factor = 1
            if has_conv_or_pool:
                cnn_inputs = tf.reshape(inputs, [batch_size, -1, network_proto.features, 1])
                shape = seq_len, network_proto.features

                layers = [cnn_inputs]
                last_num_filters = 1

                cnn_layer_index = 0
                for layer in [l for l in network_proto.layers if l.type != LayerParams.LSTM]:
                    if layer.type == LayerParams.CONVOLUTIONAL:
                        layers.append(tf.layers.conv2d(
                            name="conv2d" if cnn_layer_index == 0 else "conv2d_{}".format(cnn_layer_index),
                            inputs=layers[-1],
                            filters=layer.filters,
                            kernel_size=(layer.kernel_size.x, layer.kernel_size.y),
                            padding="same",
                            activation=tf.nn.relu,
                            reuse=reuse_variables,
                        ))
                        cnn_layer_index += 1
                        last_num_filters = layer.filters
                    elif layer.type == LayerParams.MAX_POOLING:
                        layers.append(tf.layers.max_pooling2d(
                            inputs=layers[-1],
                            pool_size=(layer.kernel_size.x, layer.kernel_size.y),
                            strides=(layer.stride.x, layer.stride.y),
                            padding="same",
                        ))

                        shape = (tf.to_int32(shape[0] // layer.stride.x),
                                 shape[1] // layer.stride.y)
                        factor *= layer.stride.x
                    else:
                        raise Exception("Unknown layer of type %s" % layer.type)

                lstm_seq_len, lstm_num_features = shape
                rnn_inputs = tf.reshape(layers[-1],
                                        [batch_size, tf.shape(layers[-1])[1],
                                         last_num_filters * lstm_num_features])

                lstm_num_features = last_num_filters * lstm_num_features
            else:
                rnn_inputs = inputs
                lstm_seq_len = seq_len
                lstm_num_features = network_proto.features

            lstm_layers = [l for l in network_proto.layers if l.type == LayerParams.LSTM]

            # Time major inputs required for lstm
            time_major_inputs = tf.transpose(rnn_inputs, [1, 0, 2])

            if len(lstm_layers) > 0:
                for i, lstm in enumerate(lstm_layers):
                    if lstm.hidden_nodes != lstm_layers[0].hidden_nodes:
                        raise Exception("Currently all lstm layers must have an equal number of hidden nodes. "
                                        "Got {} != {}".format(lstm.hidden_nodes, lstm_layers[0].hidden_nodes))

                def cpu_cudnn_compatible_lstm_backend(time_major_inputs, hidden_nodes):
                    def get_lstm_cell(num_hidden):
                        return cudnn_rnn.CudnnCompatibleLSTMCell(num_hidden, reuse=reuse_variables)

                    fw, bw = zip(*[(get_lstm_cell(hidden_nodes), get_lstm_cell(hidden_nodes)) for lstm in lstm_layers])

                    time_major_outputs, output_fw, output_bw \
                        = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(list(fw), list(bw), time_major_inputs,
                                                                         sequence_length=lstm_seq_len,
                                                                         dtype=tf.float32,
                                                                         scope="{}cudnn_lstm/stack_bidirectional_rnn".format(scope.name),
                                                                         time_major=True,
                                                                         )

                    return time_major_outputs

                def gpu_cudnn_lstm_backend(time_major_inputs, hidden_nodes):
                    # Create the Cudnn LSTM factory
                    rnn_lstm = cudnn_rnn.CudnnLSTM(len(lstm_layers), hidden_nodes,
                                                   direction='bidirectional',
                                                   kernel_initializer=tf.initializers.random_uniform(-0.1, 0.1))

                    # TODO: Check if the models are loadable from meta Graph, maybe the next line fixed this
                    rnn_lstm._saveable_cls = cudnn_rnn.CudnnLSTMSaveable

                    # Apply the lstm to the inputs
                    time_major_outputs, (output_h, output_c) = rnn_lstm(time_major_inputs)
                    return time_major_outputs

                if network_proto.backend.cudnn:
                    if gpu_enabled:
                        print("Using CUDNN LSTM backend on GPU")
                        time_major_outputs = gpu_cudnn_lstm_backend(time_major_inputs, lstm_layers[0].hidden_nodes)
                    else:
                        print("Using CUDNN compatible LSTM backend on CPU")
                        time_major_outputs = cpu_cudnn_compatible_lstm_backend(time_major_inputs, lstm_layers[0].hidden_nodes)
                else:
                    raise Exception("Only cudnn based backend supported yet.")

                # Set the output size
                output_size = lstm_layers[-1].hidden_nodes * 2
            else:
                output_size = lstm_num_features
                time_major_outputs = time_major_inputs

            # flatten to (T * N, F) for matrix multiplication. This will be reversed later
            time_major_outputs = tf.reshape(time_major_outputs, [-1, time_major_outputs.shape.as_list()[2]])

            if network_proto.dropout > 0:
                time_major_outputs = tf.nn.dropout(time_major_outputs, 1 - dropout_rate, name="dropout")

            # we need to turn off validate_shape so we can resize the variable on a codec resize
            w = tf.get_variable('W', validate_shape=False, initializer=tf.random_uniform([output_size, network_proto.classes], -0.1, 0.1))
            b = tf.get_variable('B', validate_shape=False, initializer=tf.constant(0., shape=[network_proto.classes]))

            # the output layer
            time_major_logits = tf.matmul(time_major_outputs, w) + b

            # reshape back
            time_major_logits = tf.reshape(time_major_logits, [-1, batch_size, tf.shape(w)[-1]],
                                           name="time_major_logits")

            time_major_softmax = tf.nn.softmax(time_major_logits, -1, "time_major_softmax")

            logits = tf.transpose(time_major_logits, [1, 0, 2], name="logits")
            softmax = tf.transpose(time_major_softmax, [1, 0, 2], name="softmax")

            lstm_seq_len = tf.identity(lstm_seq_len, "seq_len_out")

            # DECODER
            # ================================================================
            if network_proto.ctc == NetworkParams.CTC_DEFAULT:
                decoded, log_prob = ctc_ops.ctc_greedy_decoder(time_major_logits, lstm_seq_len, merge_repeated=network_proto.ctc_merge_repeated)
            elif network_proto.ctc == NetworkParams.CTC_FUZZY:
                decoded, log_prob = self.fuzzy_module['decoder_op'](softmax, lstm_seq_len)
            else:
                raise Exception("Unknown ctc model: '%s'. Supported are Default and Fuzzy" % network_proto.ctc)

            decoded = decoded[0]
            sparse_decoded = (
                tf.identity(decoded.indices, name="decoded_indices"),
                tf.identity(decoded.values, name="decoded_values"),
                tf.identity(decoded.dense_shape, name="decoded_shape"),
            )

            return lstm_seq_len, time_major_logits, time_major_softmax, logits, softmax, decoded, sparse_decoded, factor

    def create_placeholders(self):
        with tf.variable_scope("", reuse=False) as scope:
            inputs = tf.placeholder(tf.float32, shape=(None, None, self.network_proto.features), name="inputs")
            seq_len = tf.placeholder(tf.int32, shape=(None,), name="seq_len")
            targets = tf.sparse_placeholder(tf.int32, shape=(None, None), name="targets")
            dropout_rate = tf.placeholder(tf.float32, shape=(), name="dropout_rate")

        return inputs, seq_len, targets, dropout_rate

    def create_dataset_inputs(self, batch_size, line_height, buffer_size=1000):
        with tf.variable_scope("", reuse=False):
            def gen():
                for i, l in zip(self.raw_images, self.raw_labels):
                    if self.graph_type == "train" and len(l) == 0:
                        continue

                    yield i, l, [len(i)], [len(l)]

            def convert_to_sparse(data, labels, len_data, len_labels):
                indices = tf.where(tf.not_equal(labels, -1))
                values = tf.gather_nd(labels, indices) - 1
                shape = tf.shape(labels, out_type=tf.int64)
                return data / 255, tf.SparseTensor(indices, values, shape), len_data, len_labels

            dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32, tf.int32, tf.int32))
            if self.graph_type == "train":
                dataset = dataset.repeat().shuffle(buffer_size, seed=self.network_proto.backend.random_seed)
            else:
                pass

            dataset = dataset.padded_batch(batch_size, ([None, line_height], [None], [1], [1]),
                                           padding_values=(np.float32(0), np.int32(-1), np.int32(0), np.int32(0)))
            dataset = dataset.map(convert_to_sparse)

            data_initializer = dataset.prefetch(5).make_initializable_iterator()
            inputs = data_initializer.get_next()
            dropout_rate = tf.placeholder(tf.float32, shape=(), name="dropout_rate")
            return inputs[0], tf.reshape(inputs[2], [-1]), inputs[1], dropout_rate, data_initializer

    def create_cer(self, decoded, targets):
        # character error rate
        cer = tf.reduce_mean(tf.edit_distance(tf.cast(decoded, tf.int32), targets), name='ler')
        return cer

    def create_solver(self, targets, time_major_logits, batch_major_logits, seq_len, decoded):
        # ctc predictions
        cer = self.create_cer(decoded, targets)

        # Note for codec change: the codec size is derived upon creation, therefore the ctc ops must be created
        # using the true codec size (the W/B-Matrix may change its shape however during loading/codec change
        # to match the true codec size
        if self.network_proto.ctc == NetworkParams.CTC_DEFAULT:
            loss = ctc_ops.ctc_loss(targets,
                                    time_major_logits,
                                    seq_len,
                                    time_major=True,
                                    ctc_merge_repeated=self.network_proto.ctc_merge_repeated,
                                    ignore_longer_outputs_than_inputs=True)
        elif self.network_proto.ctc == NetworkParams.CTC_FUZZY:
            loss, deltas = self.fuzzy_module['module'].fuzzy_ctc_loss(
                batch_major_logits, targets.indices,
                targets.values,
                seq_len,
                ignore_longer_outputs_than_inputs=True)
        else:
            raise Exception("Unknown ctc model: '%s'. Supported are Default and Fuzzy" % self.network_proto.ctc)

        cost = tf.reduce_mean(loss, name='cost')
        if self.network_proto.solver == NetworkParams.MOMENTUM_SOLVER:
            optimizer = tf.train.MomentumOptimizer(self.network_proto.learning_rate, self.network_proto.momentum)
        elif self.network_proto.solver == NetworkParams.ADAM_SOLVER:
            optimizer = tf.train.AdamOptimizer(self.network_proto.learning_rate)
        else:
            raise Exception("Unknown solver of type '%s'" % self.network_proto.solver)

        gvs = optimizer.compute_gradients(cost)

        training_ops = []
        if self.network_proto.clipping_mode == NetworkParams.CLIP_NONE:
            pass
        elif self.network_proto.clipping_mode == NetworkParams.CLIP_AUTO:
            # exponentially follow the global average of gradients to set clipping
            ema = tf.train.ExponentialMovingAverage(decay=0.999)

            max_l2 = 1000
            max_grads = 1000

            grads = [grad for grad, _ in gvs]
            l2 = tf.minimum(tf.global_norm([grad for grad in grads]), max_l2)
            l2_ema_op, l2_ema = ema.apply([l2]), ema.average(l2)
            grads, _ = tf.clip_by_global_norm(grads,
                                              clip_norm=tf.minimum(l2_ema / max_l2 * max_grads, max_grads))
            gvs = zip(grads, [var for _, var in gvs])
            training_ops.append(l2_ema_op)
        elif self.network_proto.clipping_mode == NetworkParams.CLIP_CONSTANT:
            clip = self.network_proto.clipping_constant
            if clip <= 0:
                raise Exception("Invalid clipping constant. Must be greater than 0, but got {}".format(clip))

            grads = [grad for grad, _ in gvs]
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=clip)
            gvs = zip(grads, [var for _, var in gvs])
        else:
            raise Exception("Unsupported clipping mode {}".format(self.network_proto.clipping_mode))

        training_ops.append(optimizer.apply_gradients(gvs, name='grad_update_op'))
        train_op = tf.group(training_ops, name="train_op")

        return train_op, cost, cer

    def uninitialized_variables(self):
        with self.graph.as_default():
            global_vars = tf.global_variables()
            is_not_initialized = self.session.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

            return not_initialized_vars

    def reset_data(self):
        with self.graph.as_default():
            if self.data_iterator:
                self.session.run([self.data_iterator.initializer])

    def prepare(self, uninitialized_variables_only=True):
        super().prepare()
        self.reset_data()
        with self.graph.as_default():
            if uninitialized_variables_only:
                self.session.run(tf.variables_initializer(self.uninitialized_variables()))
            else:
                init_op = tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer())
                self.session.run(init_op)

    def load_weights(self, filepath, restore_only_trainable=True):
        with self.graph.as_default() as g:
            # reload trainable variables only (e. g. omitting solver specific variables)
            if restore_only_trainable:
                saver = tf.train.Saver(tf.trainable_variables())
            else:
                saver = tf.train.Saver()

            # Restore variables from disk.
            # This will possible load a weight matrix with wrong shape, thus a codec resize is necessary
            saver.restore(self.session, filepath)

    def realign_model_labels(self, indices_to_delete, indices_to_add):
        W = self.graph.get_tensor_by_name("W:0")
        B = self.graph.get_tensor_by_name("B:0")

        # removed desired entries from the data
        # IMPORTANT: Blank index is last in tensorflow but 0 in indices!
        W_val, B_val = self.session.run((W, B))
        W_val = np.delete(W_val, [i - 1 for i in indices_to_delete], axis=1)
        B_val = np.delete(B_val, [i - 1 for i in indices_to_delete], axis=0)

        # add new indices at the end
        if list(range(W_val.shape[1], W_val.shape[1] + len(indices_to_add))) != list(sorted(indices_to_add)):
            raise Exception("Additional labels must be added at the end, but got label indices {} != {}".format(
                range(W_val.shape[1], W_val.shape[1] + len(indices_to_add)), sorted(indices_to_add)))

        W_val = np.concatenate((W_val[:, :-1], np.random.uniform(-0.1, 0.1, (W_val.shape[0], len(indices_to_add))), W_val[:, -1:]), axis=1)
        B_val = np.concatenate((B_val[:-1], np.zeros((len(indices_to_add), )), B_val[-1:]), axis=0)

        # reassign values
        op_W = tf.assign(W, W_val, validate_shape=False)
        op_B = tf.assign(B, B_val, validate_shape=False)
        self.session.run((op_W, op_B))

    def save_checkpoint(self, output_file):
        with self.graph.as_default() as g:
            saver = tf.train.Saver()
            saver.save(self.session, output_file)

    def train_batch(self, x, len_x, y):
        out = self.session.run(
            [self.loss, self.train_op, self.logits, self.output_seq_len, self.cer, self.decoded],
            feed_dict={
                self.inputs: x,
                self.input_seq_len: len_x,
                self.targets: y,
                self.dropout_rate: self.network_proto.dropout,
            }
        )

        if np.isfinite(out[0]):
            # only update gradients if finite loss
            self.session.run(
                [self.train_op],
                feed_dict={
                    self.inputs: x,
                    self.input_seq_len: len_x,
                    self.targets: y,
                    self.dropout_rate: self.network_proto.dropout,
                }
            )
        else:
            print("WARNING: Infinite loss. Skipping batch.", file=sys.stderr)

        return out

    def train_dataset(self):
        out = self.session.run(
            [self.loss, self.softmax, self.output_seq_len, self.cer, self.decoded, self.targets],
            feed_dict={
                self.dropout_rate: self.network_proto.dropout,
            }
        )

        if np.isfinite(out[0]):
            # only update gradients if finite loss
            self.session.run(
                [self.train_op],
                feed_dict={
                    self.dropout_rate: self.network_proto.dropout,
                }
            )
        else:
            print("WARNING: Infinite loss. Skipping batch.", file=sys.stderr)

        return out

    def predict_raw(self, x, len_x):
        return self.session.run(
            [self.softmax, self.output_seq_len, self.decoded],
            feed_dict={
                self.inputs: x,
                self.input_seq_len: len_x,
                self.dropout_rate: 0,
            })

    def predict_dataset(self):
        return self.session.run(
            [self.softmax, self.output_seq_len, self.decoded],
            feed_dict={
                self.dropout_rate: 0,
            })

    def train(self, batch_x, batch_y):
        if batch_x and batch_y:
            x, len_x = TensorflowModel.__sparse_data_to_dense(batch_x)
            y = TensorflowModel.__to_sparse_matrix(batch_y)

            cost, probs, seq_len, ler, decoded = self.train_batch(x, len_x, y)
            gt = batch_y
        else:
            cost, probs, seq_len, ler, decoded, gt = self.train_dataset()
            gt = TensorflowModel.__sparse_to_lists(gt)

        probs = np.roll(probs, 1, axis=2)
        return {
            "loss": cost,
            "probabilities": probs,
            "ler": ler,
            "decoded": TensorflowModel.__sparse_to_lists(decoded),
            "gt": gt,
            "logits_lengths": seq_len,
        }

    def predict(self):
        try:
            while True:
                probs, seq_len, decoded = self.predict_dataset()
                probs = np.roll(probs, 1, axis=2)
                # decoded = TensorflowBackend.__sparse_to_lists(decoded)
                for l, s in zip(probs, seq_len):
                    yield self.ctc_decoder.decode(l[:s])

        except tf.errors.OutOfRangeError as e:
            # no more data available
            pass

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

    def output_to_input_position(self, x):
        return x * self.scale_factor
