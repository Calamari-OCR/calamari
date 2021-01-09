import tensorflow as tf
from typing import Dict, Type, List, Tuple, Any
import bidi.algorithm as bidi
import Levenshtein

from tfaip.base.model.modelbase import ModelBase, ModelBaseParams
from tfaip.util.typing import AnyNumpy

from calamari_ocr.ocr.model.graph import Graph
from calamari_ocr.ocr.model.params import ModelParams
from tensorflow.python.ops import math_ops

from calamari_ocr.ocr.predict.params import Prediction

keras = tf.keras
K = keras.backend
KL = keras.layers


class Model(ModelBase):
    @staticmethod
    def get_params_cls() -> Type[ModelBaseParams]:
        return ModelParams

    @classmethod
    def _get_additional_layers(cls) -> List[Type[tf.keras.layers.Layer]]:
        return [Graph]

    def __init__(self, params: ModelParams):
        super(Model, self).__init__(params)
        self._params: ModelParams = params

    def _best_logging_settings(self) -> Tuple[str, str]:
        return "min", "CER"

    def create_graph(self, params: ModelBaseParams) -> 'GraphBase':
        return Graph(params)

    def _loss(self, inputs: Dict[str, tf.Tensor], outputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        def to_2d_list(x):
            return K.expand_dims(K.flatten(x), axis=-1)

        # note: blank is last index
        loss = KL.Lambda(lambda args: K.ctc_batch_cost(args[0] - 1, args[1], args[2], args[3]), name='ctc')(
            (inputs['gt'], outputs['blank_last_softmax'], to_2d_list(outputs['out_len']), to_2d_list(inputs['gt_len'])))
        return {
            'loss': loss
        }

    def _extended_metric(self, inputs: Dict[str, tf.Tensor], outputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        def create_cer(decoded, targets, targets_length):
            greedy_decoded = tf.sparse.from_dense(decoded)
            sparse_targets = tf.cast(K.ctc_label_dense_to_sparse(targets, math_ops.cast(
                K.flatten(targets_length), dtype='int32')), 'int32')
            return tf.edit_distance(tf.cast(greedy_decoded, tf.int32), sparse_targets, normalize=True)

        # Note for codec change: the codec size is derived upon creation, therefore the ctc ops must be created
        # using the true codec size (the W/B-Matrix may change its shape however during loading/codec change
        # to match the true codec size
        cer = KL.Lambda(lambda args: create_cer(*args), output_shape=(1,), name='cer')((outputs['decoded'], inputs['gt'], inputs['gt_len']))
        return {
            'CER': cer,
        }

    def _sample_weights(self, inputs: Dict[str, tf.Tensor], targets: Dict[str, tf.Tensor]) -> Dict[str, Any]:
        return {
            "CER": K.flatten(targets['gt_len']),
        }

    def print_evaluate(self, inputs: Dict[str, AnyNumpy], outputs: Prediction, targets: Dict[str, AnyNumpy],
                       data: 'CalamariData', print_fn):
        pred_sentence = outputs.sentence
        gt_sentence = targets['sentence']
        lr = "\u202A\u202B"
        cer = Levenshtein.distance(pred_sentence, gt_sentence) / len(gt_sentence)
        print_fn("\n  CER:  {}".format(cer) +
                 "\n  PRED: '{}{}{}'".format(lr[bidi.get_base_level(pred_sentence)], pred_sentence, "\u202C") +
                 "\n  TRUE: '{}{}{}'".format(lr[bidi.get_base_level(gt_sentence)], gt_sentence, "\u202C"))
