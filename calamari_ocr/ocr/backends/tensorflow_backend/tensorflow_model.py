import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.nn_grad import _BroadcastMul
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, DropoutWrapper, LSTMBlockFusedCell, LSTMBlockCell
import tensorflow.contrib.cudnn_rnn as cudnn_rnn
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import ctc_ops
import numpy as np

from calamari_ocr.proto import LayerParams, NetworkParams


class TensorflowModel:

    @staticmethod
    def load(network_proto, restore):
        try:
            filename = restore
            graph = tf.Graph()
            with graph.as_default() as g:
                session = tf.Session(graph=graph,
                                     config=tf.ConfigProto(intra_op_parallelism_threads=0,
                                                           inter_op_parallelism_threads=0,
                                                           #session_inter_op_thread_pool=[{'num_threads': threads}],
                                                           #use_per_session_threads=True,
                                                           ))
                with tf.variable_scope("", reuse=False) as scope:

                    saver = tf.train.import_meta_graph(filename + '.meta')
                    saver.restore(session, filename)

                    inputs = g.get_tensor_by_name("inputs:0")
                    seq_len = g.get_tensor_by_name("seq_len:0")
                    try:
                        seq_len_out = g.get_tensor_by_name("seq_len_out:0")
                    except:
                        print("loaded old model!")
                        seq_len_out = seq_len / 4

                    try:
                        dropout_rate = g.get_tensor_by_name("dropout_rate:0")
                    except:
                        print("loaded old model without dropout rate")
                        dropout_rate = tf.placeholder(tf.float32, shape=(), name="dropout_rate")

                    targets = tf.SparseTensor(
                        g.get_tensor_by_name("targets/indices:0"),
                        g.get_tensor_by_name("targets/values:0"),
                        g.get_tensor_by_name("targets/shape:0"))
                    cost = g.get_tensor_by_name("cost:0")
                    train_op = g.get_operation_by_name('train_op')
                    ler = g.get_tensor_by_name("ler:0")
                    decoded = (
                        g.get_tensor_by_name("decoded_indices:0"),
                        g.get_tensor_by_name("decoded_values:0"),
                        g.get_tensor_by_name("decoded_shape:0")
                    )
                    logits = g.get_tensor_by_name("softmax:0")

                    return TensorflowModel(network_proto, graph, session, inputs, seq_len, seq_len_out, targets, train_op, cost, ler, decoded, logits, dropout_rate)
        except KeyError as e:
            # TODO: Crash if cudnn is loaded
            # Workaround create new graph and load weights
            print("Attempting a workaround: New graph and load weights")

            model = TensorflowModel.from_proto(network_proto)
            with model.graph.as_default() as g:
                try:
                    saver = tf.train.Saver()
                    saver.restore(model.session, restore)
                except tf.errors.NotFoundError as e:
                    print("Attempting workaround: only loading trainable variables")
                    saver = tf.train.Saver(tf.trainable_variables())
                    saver.restore(model.session, restore)

            return model

    @staticmethod
    def from_proto(network_proto):
        reuse_variables = False
        intra_threads = 0
        inter_threads = 0

        # load fuzzy ctc module if available
        if len(network_proto.backend.fuzzy_ctc_library_path) > 0 and network_proto.ctc == NetworkParams.CTC_FUZZY:
            from calamari_ocr.ocr.backends.tensorflow_backend.tensorflow_fuzzy_ctc_loader import load as load_fuzzy
            fuzzy_module = load_fuzzy(network_proto.backend.fuzzy_ctc_library_path)
        else:
            fuzzy_module = None

        graph = tf.Graph()
        with graph.as_default():
            tf.set_random_seed(network_proto.backend.random_seed)
            session = tf.Session(graph=graph,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=intra_threads,
                                                       inter_op_parallelism_threads=inter_threads,
                                                       ))
            gpu_enabled = False
            for d in session.list_devices():
                if d.device_type == "GPU":
                    gpu_enabled = True
                    break

            inputs = tf.placeholder(tf.float32, shape=(None, None, network_proto.features), name="inputs")
            batch_size = tf.shape(inputs)[0]
            seq_len = tf.placeholder(tf.int32, shape=(None,), name="seq_len")
            targets = tf.sparse_placeholder(tf.int32, shape=(None, None), name="targets")
            dropout_rate = tf.placeholder(tf.float32, shape=(), name="dropout_rate")

            with tf.variable_scope("", reuse=reuse_variables) as scope:
                no_layers = len(network_proto.layers) == 0
                if not no_layers:
                    has_conv_or_pool = network_proto.layers[0].type != LayerParams.LSTM
                else:
                    has_conv_or_pool = False

                if has_conv_or_pool:
                    cnn_inputs = tf.reshape(inputs, [batch_size, -1, network_proto.features, 1])
                    shape = seq_len, network_proto.features

                    layers = [cnn_inputs]
                    last_num_filters = 1

                    for layer in [l for l in network_proto.layers if l.type != LayerParams.LSTM]:
                        if layer.type == LayerParams.CONVOLUTIONAL:
                            layers.append(tf.layers.conv2d(
                                inputs=layers[-1],
                                filters=layer.filters,
                                kernel_size=(layer.kernel_size.x, layer.kernel_size.y),
                                padding="same",
                                activation=tf.nn.relu,
                            ))
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
                W = tf.get_variable('W', validate_shape=False, initializer=tf.random_uniform([output_size, network_proto.classes], -0.1, 0.1))
                b = tf.get_variable('B', validate_shape=False, initializer=tf.constant(0., shape=[network_proto.classes]))

                # the output layer
                time_major_logits = tf.matmul(time_major_outputs, W) + b

                # reshape back
                time_major_logits = tf.reshape(time_major_logits, [-1, batch_size, tf.shape(W)[-1]], name="time_major_logits")

                time_major_softmax = tf.nn.softmax(time_major_logits, -1, "time_major_softmax")

                logits = tf.transpose(time_major_logits, [1, 0, 2], name="logits")
                softmax = tf.transpose(time_major_softmax, [1, 0, 2], name="softmax")

                # ctc predictions
                # Note for codec change: the codec size is derived upon creation, therefore the ctc ops must be created
                # using the true codec size (the W/B-Matrix may change its shape however during loading/codec change
                # to match the true codec size
                if network_proto.ctc == NetworkParams.CTC_DEFAULT:
                    loss = ctc_ops.ctc_loss(targets,
                                            time_major_logits,
                                            lstm_seq_len, time_major=True, ctc_merge_repeated=network_proto.ctc_merge_repeated, ignore_longer_outputs_than_inputs=True)
                    decoded, log_prob = ctc_ops.ctc_greedy_decoder(time_major_logits, lstm_seq_len, merge_repeated=network_proto.ctc_merge_repeated)
                    # decoded, log_prob = ctc_ops.ctc_beam_search_decoder(time_major_logits, lstm_seq_len, merge_repeated=model_settings["merge_repeated"])
                elif network_proto.ctc == NetworkParams.CTC_FUZZY:
                    loss, deltas = fuzzy_module['module'].fuzzy_ctc_loss(logits, targets.indices, targets.values, lstm_seq_len)
                    decoded, log_prob = fuzzy_module['decoder_op'](softmax, lstm_seq_len)
                else:
                    raise Exception("Unknown ctc model: '%s'. Supported are Default and Fuzzy" % network_proto.ctc)

                decoded = decoded[0]
                sparse_decoded = (
                    tf.identity(decoded.indices, name="decoded_indices"),
                    tf.identity(decoded.values, name="decoded_values"),
                    tf.identity(decoded.dense_shape, name="decoded_shape"),
                )

                cost = tf.reduce_mean(loss, name='cost')
                if network_proto.solver == NetworkParams.MOMENTUM_SOLVER:
                    optimizer = tf.train.MomentumOptimizer(network_proto.learning_rate, network_proto.momentum)
                elif network_proto.solver == NetworkParams.ADAM_SOLVER:
                    optimizer = tf.train.AdamOptimizer(network_proto.learning_rate)
                else:
                    raise Exception("Unknown solver of type '%s'" % network_proto.solver)

                gvs = optimizer.compute_gradients(cost)

                # exponentially follow the gradients to set clipping values
                ema = tf.train.ExponentialMovingAverage(decay=0.99)

                def get_ema_ops(a):
                    l2 = tf.nn.l2_loss(a)
                    return ema.apply([l2]), ema.average(l2)

                means = [get_ema_ops(grad) for grad, var in gvs]
                # maybe follow values instead of grads
                gvs = [(tf.clip_by_value(grad, -avg * 10, avg * 10), var) for (grad, var), (m, avg) in zip(gvs, means)]

                train_op = optimizer.apply_gradients(gvs, name='grad_update_op')

                # resulting operation for training (grad update + ema update)
                train_op = tf.group([train_op] + [m for m, _ in means])

                ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded, tf.int32), targets), name='ler')

                lstm_seq_len = tf.identity(lstm_seq_len, "seq_len_out")

                return TensorflowModel(network_proto, graph, session, inputs, seq_len, lstm_seq_len,
                                       targets, train_op, cost, ler, sparse_decoded,
                                       softmax, dropout_rate)

    def __init__(self,
                 network_proto,
                 graph,
                 session,
                 inputs,
                 seq_len,
                 seq_len_out,
                 targets,
                 optimizer,
                 cost,
                 ler,
                 decoded,
                 logits,
                 dropout_rate
                 ):
        self.network_proto = network_proto
        self.graph = graph
        self.session = session
        self.inputs = inputs
        self.seq_len = seq_len
        self.seq_len_out = seq_len_out
        self.targets = targets
        self.optimizer = optimizer
        self.cost = cost
        self.ler = ler
        self.decoded = decoded
        self.logits = logits
        self.dropout_rate = dropout_rate

    def uninitialized_variables(self):
        with self.graph.as_default():
            global_vars = tf.global_variables()
            is_not_initialized = self.session.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

            return not_initialized_vars

    def prepare(self, train, uninitialized_variables_only=True):
        if train:
            with self.graph.as_default():
                if uninitialized_variables_only:
                    self.session.run(tf.variables_initializer(self.uninitialized_variables()))
                else:
                    init_op = tf.group(tf.global_variables_initializer(),
                                       tf.local_variables_initializer())
                    self.session.run(init_op)

    def load_weights(self, model_file):
        with self.graph.as_default() as g:
            # reload trainable variables only (e. g. omitting solver specific variables)
            saver = tf.train.Saver(tf.trainable_variables())

            # Restore variables from disk.
            # This will possible load a weight matrix with wrong shape, thus a codec resize is necessary
            saver.restore(self.session, model_file)

    def realign_labels(self, indices_to_delete, indices_to_add):
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

    def save(self, output_file):
        with self.graph.as_default() as g:
            saver = tf.train.Saver()
            saver.save(self.session, output_file)

    def train(self, x, len_x, y):
        return self.session.run(
            [self.cost, self.optimizer, self.logits, self.seq_len_out, self.ler, self.decoded],
            feed_dict={
                self.inputs: x,
                self.seq_len: len_x,
                self.targets: y,
                self.dropout_rate: self.network_proto.dropout,
            }
        )

    def predict(self, x, len_x):
        return self.session.run(
            [self.logits, self.seq_len_out, self.decoded],
            feed_dict={
                self.inputs: x,
                self.seq_len: len_x,
                self.dropout_rate: 0,
            })