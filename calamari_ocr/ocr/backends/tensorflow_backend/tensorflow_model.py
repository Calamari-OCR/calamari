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
            print(e)
            print("Attempting a workaround")

            model = TensorflowModel.from_proto(network_proto)
            with model.graph.as_default() as g:
                saver = tf.train.Saver()
                saver.restore(model.session, restore)

            return model

    @staticmethod
    def from_proto(network_proto):
        reuse_variables = False
        intra_threads = 0
        inter_threads = 0

        graph = tf.Graph()
        with graph.as_default():
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

                W = tf.get_variable('W', initializer=tf.random_uniform([output_size, network_proto.classes], -0.1, 0.1))
                b = tf.get_variable('B', initializer=tf.constant(0., shape=[network_proto.classes]))

                time_major_logits = tf.matmul(time_major_outputs, W) + b

                # reshape back
                time_major_logits = tf.reshape(time_major_logits, [-1, batch_size, network_proto.classes], name="time_major_logits")

                time_major_softmax = tf.nn.softmax(time_major_logits, -1, "time_major_softmax")

                logits = tf.transpose(time_major_logits, [1, 0, 2], name="logits")
                softmax = tf.transpose(time_major_softmax, [1, 0, 2], name="softmax")

                # ctc predictions
                if network_proto.ctc == NetworkParams.CTC_DEFAULT:
                    loss = ctc_ops.ctc_loss(targets,
                                            time_major_logits,
                                            lstm_seq_len, time_major=True, ctc_merge_repeated=network_proto.ctc_merge_repeated, ignore_longer_outputs_than_inputs=True)
                    decoded, log_prob = ctc_ops.ctc_greedy_decoder(time_major_logits, lstm_seq_len, merge_repeated=network_proto.ctc_merge_repeated)
                    # decoded, log_prob = ctc_ops.ctc_beam_search_decoder(time_major_logits, lstm_seq_len, merge_repeated=model_settings["merge_repeated"])
                elif network_proto.ctc == NetworkParams.CTC_FUZZY:
                    raise Exception("The fuzzy decoder is not supported yet!")
                    # loss, deltas = fuzzy_module.fuzzy_ctc_loss(logits, targets.indices, targets.values, lstm_seq_len)
                    # decoded, log_prob = fuzzy_ctc_greedy_decoder(softmax, lstm_seq_len)
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
                train_op = optimizer.apply_gradients(gvs, name='train_op')

                ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded, tf.int32), targets), name='ler')

                init_op = tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer())
                session.run(init_op)

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

    def load_weights(self, model_file):
        with self.graph.as_default() as g:
            # TODO: maybe use trainable_variables
            all_var_names = [v for v in tf.global_variables()
                             if not v.name.startswith("W:") and not v.name.startswith("B:")
                             and "Adam" not in v.name and "beta1_power" not in v.name and "beta2_power" not in v.name]
            saver = tf.train.Saver(all_var_names)

            # Restore variables from disk.
            saver.restore(self.session, model_file)

    def save(self, output_file):
        with self.graph.as_default() as g:
            saver = tf.train.Saver()
            saver.save(self.session, output_file)

    def train(self, x, len_x, y):
        return self.session.run(
            [self.cost, self.optimizer, self.logits, self.ler, self.decoded],
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