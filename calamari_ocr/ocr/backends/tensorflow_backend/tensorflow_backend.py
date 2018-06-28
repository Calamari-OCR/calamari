import numpy as np
import tensorflow as tf

from calamari_ocr.ocr.backends.backend_interface import BackendInterface
from calamari_ocr.ocr.backends.tensorflow_backend.tensorflow_model import TensorflowModel


class TensorflowBackend(BackendInterface):
    def __init__(self,
                 network_proto,
                 restore,
                 weights,
                 processes=-1):
        super().__init__(network_proto)
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph,
                                  config=tf.ConfigProto(
                                      intra_op_parallelism_threads=processes if processes >= 0 else network_proto.backend.num_intra_threads,
                                      inter_op_parallelism_threads=processes if processes >= 0 else network_proto.backend.num_inter_threads,
                                  ))
        self.restore = restore
        self.weights = weights
        self.first_model = True

    def create_net(self, restore, weights, graph_type, batch_size=-1):
        model = TensorflowModel(self.network_proto, self.graph, self.session, graph_type, batch_size, reuse_weights=not self.first_model)
        self.first_model = False
        if weights:
            model.load_weights(weights, restore_only_trainable=True)

        if restore:
            try:
                model.load_weights(restore, restore_only_trainable=False)
            except tf.errors.NotFoundError as e:
                if "opaque_kernel" in e.message:
                    print(e)
                    raise Exception("This exception probabily occurred when loading a CPU model on the GPU. This is currently not supported by TensorFlow")

                # this might be cudnn related, try again, but skip non trainable and opaque kernel
                with self.graph.as_default():
                    saver = tf.train.Saver(tf.trainable_variables())
                    saver.restore(self.session, restore)

        return model


