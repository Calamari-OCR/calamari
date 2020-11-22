import tensorflow as tf
import numpy as np
from typing import Dict, Type, List, Tuple, Any
import bidi.algorithm as bidi
import Levenshtein

from tfaip.base.model import ModelBase, GraphBase, ModelBaseParams
from tfaip.util.typing import AnyNumpy

from calamari_ocr.ocr.model.graph import CalamariGraph
from calamari_ocr.ocr.model.params import ModelParams
from tensorflow.python.ops import math_ops

keras = tf.keras
K = keras.backend
KL = keras.layers
Model = keras.Model


class CalamariModel(ModelBase):
    @staticmethod
    def get_params_cls() -> Type[ModelBaseParams]:
        return ModelParams

    @classmethod
    def _get_additional_layers(cls) -> List[Type[tf.keras.layers.Layer]]:
        return [CalamariGraph]

    def __init__(self, params: ModelParams):
        super(CalamariModel, self).__init__(params)
        self._params: ModelParams = params

    def _best_logging_settings(self) -> Tuple[str, str]:
        return "min", "cer_metric"

    def create_graph(self, params: ModelBaseParams) -> 'GraphBase':
        return CalamariGraph(params)

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

    def _target_prediction(self,
                           targets: Dict[str, AnyNumpy],
                           outputs: Dict[str, AnyNumpy],
                           data: 'CalamariData',
                           ) -> Tuple[Any, Any]:
        return targets['gt'], outputs['decoded'][np.where(outputs['decoded'] != -1)]

    def _print_evaluate(self, inputs: Dict[str, AnyNumpy], outputs: Dict[str, AnyNumpy], targets: Dict[str, AnyNumpy],
                        data: 'CalamariData', print_fn):
        gt, pred = self._target_prediction(targets, outputs, data)
        pred_sentence = data.params().text_post_processor.apply("".join(data.params().codec.decode(pred)))
        gt_sentence = data.params().text_post_processor.apply("".join(data.params().codec.decode(gt)))
        lr = "\u202A\u202B"
        cer = Levenshtein.distance(pred_sentence, gt_sentence) / len(gt_sentence)
        print_fn("\n  CER:  {}".format(cer) +
                 "\n  PRED: '{}{}{}'".format(lr[bidi.get_base_level(pred_sentence)], pred_sentence, "\u202C") +
                 "\n  TRUE: '{}{}{}'".format(lr[bidi.get_base_level(gt_sentence)], gt_sentence, "\u202C"))
