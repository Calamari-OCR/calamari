import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.nn_grad import _BroadcastMul
from tensorflow.python.framework import sparse_tensor


def load(library_path):
    fuzzy_module = tf.load_op_library(library_path)

    @ops.RegisterGradient("FuzzyCTCLoss")
    def _FuzzyCTCLossGrad(op, grad_loss, _):
        grad_without_gradient = array_ops.prevent_gradient(
            op.outputs[1], message="Currently there is no way to take the second "
                                   " derivative of ctc_loss due to the fused implementation's interaction "
                                   " with tf.gradients()")
        return [_BroadcastMul(tf.expand_dims(grad_loss, -1), grad_without_gradient), None, None, None]

    def fuzzy_ctc_greedy_decoder(inputs, sequence_length):
        outputs = fuzzy_module.fuzzy_ctc_greedy_decoder(inputs, sequence_length)
        (decoded_ix, decoded_val, decoded_shape, log_probabilities) = outputs
        return ([sparse_tensor.SparseTensor(decoded_ix, decoded_val, decoded_shape)],
                log_probabilities)

    return {"module": fuzzy_module, "decoder_op": fuzzy_ctc_greedy_decoder}

