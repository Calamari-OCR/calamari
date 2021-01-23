import tensorflow as tf
from typing import Dict, Type, List, Tuple, Any
import bidi.algorithm as bidi
import Levenshtein

from tfaip.base.model.modelbase import ModelBase, ModelBaseParams
from tfaip.util.typing import AnyNumpy

from calamari_ocr.ocr.model.params import ModelParams
from tensorflow.python.ops import math_ops

from calamari_ocr.ocr.model.ensemblegraph import EnsembleGraph
from calamari_ocr.ocr.predict.params import Prediction

keras = tf.keras
K = keras.backend
KL = keras.layers


class EnsembleModel(ModelBase):
    @staticmethod
    def get_params_cls() -> Type[ModelBaseParams]:
        return ModelParams

    @classmethod
    def _get_additional_layers(cls) -> List[Type[tf.keras.layers.Layer]]:
        return [EnsembleGraph]

    def __init__(self, params: ModelParams):
        super(EnsembleModel, self).__init__(params)
        assert(params.ensemble is not None)  # Voter variable not set
        assert(params.ensemble > 1)  # At least 2 voters required
        self._params: ModelParams = params

    def _best_logging_settings(self) -> Tuple[str, str]:
        return "min", "CER"

    def create_graph(self, params: ModelBaseParams) -> 'GraphBase':
        return EnsembleGraph(params)

    def _loss(self, inputs: Dict[str, tf.Tensor], outputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        def to_2d_list(x):
            return K.expand_dims(K.flatten(x), axis=-1)

        # note: blank is last index
        loss = KL.Lambda(lambda args: K.ctc_batch_cost(args[0] - 1, args[1], args[2], args[3]), name='ctc')(
            (inputs['gt'], outputs['blank_last_softmax'], to_2d_list(outputs['out_len']), to_2d_list(inputs['gt_len'])))
        losses = {
            'loss': loss
        }

        for i in range(self._params.ensemble):
            loss = KL.Lambda(lambda args: K.ctc_batch_cost(args[0] - 1, args[1], args[2], args[3]), name=f'ctc_{i}')(
                (inputs['gt'], outputs[f'blank_last_softmax_{i}'], to_2d_list(outputs[f'out_len_{i}']),
                 to_2d_list(inputs['gt_len'])))
            losses[f'loss_{i}'] = loss

        return losses

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
        metrics = {
            'CER': cer,
        }

        for i in range(self._params.ensemble):
            cer = KL.Lambda(lambda args: create_cer(*args), output_shape=(1,), name=f'cer_{i}')(
                (outputs[f'decoded_{i}'], inputs['gt'], inputs['gt_len']))
            metrics[f"CER_{i}"] = cer

        return metrics

    def _sample_weights(self, inputs: Dict[str, tf.Tensor], targets: Dict[str, tf.Tensor]) -> Dict[str, Any]:
        weights = {
            "CER": K.flatten(targets['gt_len']),
        }
        for i in range(self._params.ensemble):
            # Only count CERs of this voter for validation
            weights[f"CER_{i}"] = weights["CER"] * tf.cast(tf.equal(K.flatten(targets['fold_id']), i), tf.int32)

        return weights

    def print_evaluate(self, inputs: Dict[str, AnyNumpy], outputs: Prediction, targets: Dict[str, AnyNumpy],
                       data, print_fn=print):
        gt_sentence = targets['sentence']
        lr = "\u202A\u202B"
        s = ""

        pred_sentence = outputs.sentence
        cer = Levenshtein.distance(pred_sentence, gt_sentence) / len(gt_sentence)
        s += (
                "\n  PRED (CER={:.2f}): '{}{}{}'".format(cer, lr[bidi.get_base_level(pred_sentence)], pred_sentence, "\u202C") +
                "\n  TRUE:            '{}{}{}'".format(lr[bidi.get_base_level(gt_sentence)], gt_sentence, "\u202C"))

        print_fn(s)
